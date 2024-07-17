from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import struct
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Literal,
    NamedTuple,
    Type,
    get_args,
)
from uuid import UUID, uuid4

import numpy as np
import pyrocko.moment_tensor as pmt
from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    PositiveFloat,
    PrivateAttr,
    ValidationError,
)
from pyrocko import gf
from pyrocko.guts import Float
from pyrocko.trace import FrequencyResponse
from typing_extensions import Self

from qseek.stats import PROGRESS
from qseek.utils import (
    ChannelSelector,
    ChannelSelectors,
    MeasurementUnit,
    Range,
    human_readable_bytes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pyrocko.cake import LayeredModel
    from pyrocko.trace import Trace

KM = 1e3
NM = 1e-9

logger = logging.getLogger(__name__)

PeakAmplitude = Literal["horizontal", "vertical", "absolute"]
GFInterpolation = Literal["nearest_neighbor", "multilinear"]

Components = Literal["Z", "R", "T"]
_DIPS: dict[Components, float] = {"Z": -90.0, "R": 0.0, "T": 0.0}
_AZIMUTHS: dict[Components, float] = {"Z": 0.0, "R": 0.0, "T": 90.0}


class BruneResponse(FrequencyResponse):
    duration = Float.T()

    def evaluate(self, freqs):
        return 1.0 / (1.0 + (freqs * self.duration) ** 2)


class MTSourceCircularCrack(gf.MTSource):
    duration = Float.T()
    stress_drop = Float.T()
    radius = Float.T()
    magnitude = Float.T()


def _get_target(targets: list[gf.Target], nsl: tuple[str, str, str]) -> gf.Target:
    """Get the target from the list of targets based on the given NSL codes.

    Args:
        targets (list[gf.Target]): List of targets to search from.
        nsl (tuple[str, str, str]): Network, station, and location codes.

    Returns:
        gf.Target: The target matching the given NSL codes.

    Raises:
        KeyError: If no target is found for the given NSL codes.
    """
    for target in targets:
        if nsl == target.codes[:3]:
            return target
    raise KeyError(f"No target for {nsl}.")


def trace_amplitude(traces: list[Trace], channel_selector: ChannelSelector) -> float:
    """Normalize traces channels.

    Args:
        traces (list[Trace]): A list of traces to normalize.
        channel_selector (ChannelSelector): The channel selector to use.

    Returns:
        Trace: The normalized trace.

    Raises:
        KeyError: If there are no traces to normalize.
    """
    trace_selection = channel_selector(traces)
    if not trace_selection:
        raise KeyError("No traces to normalize.")
    data = np.array([tr.ydata for tr in trace_selection])
    data = np.linalg.norm(np.atleast_2d(data), axis=0)
    return float(data.max())


class PeakAmplitudesBase(BaseModel):
    gf_store_id: str = Field(
        default="moment_magnitude",
        description="Pyrocko Store ID for peak amplitude models.",
    )
    quantity: MeasurementUnit = Field(
        default="displacement",
        description="Quantity for the peak amplitude.",
    )
    frequency_range: Range | None = Field(
        default=None,
        description="Frequency range for the peak amplitude.",
    )
    max_distance: PositiveFloat = Field(
        default=100.0 * KM,
        description="Maximum surface distances to the source for the receivers.",
    )
    source_depth_delta: PositiveFloat = Field(
        default=1.0 * KM,
        description="Source depth increment for the peak amplitude models.",
    )
    reference_magnitude: float = Field(
        default=1.0,
        ge=-1.0,
        le=8.0,
        description="Reference moment magnitude in Mw.",
    )
    rupture_velocities: Range = Field(
        default=Range(0.8, 0.9),
        description="Rupture velocity range as fraction of the shear wave velocity.",
    )
    stress_drop: Range = Field(
        default=Range(1.0e6, 10.0e6),
        description="Stress drop range in Pa.",
    )
    gf_interpolation: GFInterpolation = Field(
        default="multilinear",
        description="Interpolation method for the Pyrocko GF Store.",
    )


class SiteAmplitude(NamedTuple):
    magnitude: float
    distance_epi: float
    peak_horizontal: float
    peak_vertical: float
    peak_absolute: float

    @classmethod
    def from_traces(
        cls,
        receiver: gf.Receiver,
        traces: list[Trace],
        magnitude: float,
    ) -> Self:
        return cls(
            magnitude=magnitude,
            distance_epi=np.sqrt(receiver.north_shift**2 + receiver.east_shift**2),
            peak_horizontal=trace_amplitude(traces, ChannelSelectors.Horizontal),
            peak_vertical=trace_amplitude(traces, ChannelSelectors.Vertical),
            peak_absolute=trace_amplitude(traces, ChannelSelectors.All),
        )


class ModelledAmplitude(NamedTuple):
    magnitude: float
    quantity: MeasurementUnit
    peak_amplitude: PeakAmplitude
    distance_epi: float
    average: float
    median: float
    std: float
    mad: float

    def combine(
        self,
        other: ModelledAmplitude,
        weight: float = 1.0,
    ) -> ModelledAmplitude:
        """Combines with another ModelledAmplitude using a weighted average.

        Args:
            other (ModelledAmplitude): The ModelledAmplitude to be combined with.
            weight (float, optional): The weight of the amplitude being combined.
                Defaults to 1.0.

        Returns:
            Self: A new instance of the ModelledAmplitude class with the combined values.

        Raises:
            ValueError: If the weight is not between 0.0 and 1.0 (inclusive).
            ValueError: If the distances of the amplitudes are different.
            ValueError: If the peak amplitudes of the amplitudes are different.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Invalid weight {weight}. Must be between 0.0 and 1.0.")
        if self.distance_epi != other.distance_epi:
            raise ValueError("Cannot add amplitudes with different distances")
        if self.quantity != other.quantity:
            raise ValueError("Cannot add amplitudes with different quantities ")
        if self.magnitude != other.magnitude:
            raise ValueError("Cannot add amplitudes with different reference magnitude")
        if self.peak_amplitude != other.peak_amplitude:
            raise ValueError("Cannot add amplitudes with different peak amplitudes ")
        rcp_weight = 1.0 - weight
        return ModelledAmplitude(
            magnitude=self.magnitude,
            peak_amplitude=self.peak_amplitude,
            quantity=self.quantity,
            distance_epi=self.distance_epi,
            average=self.average * rcp_weight + other.average * weight,
            median=self.median * rcp_weight + other.median * weight,
            std=self.std * rcp_weight + other.std * weight,
            mad=self.mad * rcp_weight + other.mad * weight,
        )

    def estimate_magnitude(self, observed_amplitude: float) -> float:
        """Get the moment magnitude for the given observed amplitude.

        Args:
            observed_amplitude (float): The observed amplitude.

        Returns:
            float: The moment magnitude.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.magnitude + np.log10(observed_amplitude / self.average)


class SiteAmplitudesCollection(BaseModel):
    source_depth: float
    quantity: MeasurementUnit
    rupture_velocities: Range
    stress_drop: Range
    gf_store_id: str
    frequency_range: Range

    site_amplitudes: list[SiteAmplitude] = Field(default_factory=list)

    @staticmethod
    def _get_numpy_array(attribute: str) -> Callable:
        def wrapped(self) -> np.ndarray:
            return np.array([getattr(sa, attribute) for sa in self.site_amplitudes])

        return wrapped

    _distances = cached_property[np.ndarray](_get_numpy_array("distance_epi"))
    _magnitudes = cached_property[np.ndarray](_get_numpy_array("magnitude"))
    _vertical = cached_property[np.ndarray](_get_numpy_array("peak_vertical"))
    _absolute = cached_property[np.ndarray](_get_numpy_array("peak_absolute"))
    _horizontal = cached_property[np.ndarray](_get_numpy_array("peak_horizontal"))

    def _clear_cache(self) -> None:
        keys = {"_distances", "_horizontal", "_vertical", "_absolute", "_magnitudes"}
        for key in keys:
            self.__dict__.pop(key, None)

    def get_amplitude_model(
        self,
        distance: float,
        n_amplitudes: int,
        distance_cutoff: float = 0.0,
        reference_magnitude: float = 1.0,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> ModelledAmplitude:
        """Get the amplitudes for a given distance.

        Args:
            distance (float): The epicentral distance to retrieve the amplitudes for.
            n_amplitudes (int): The number of amplitudes to retrieve.
            distance_cutoff (float): The maximum distance allowed for
                the retrieved amplitudes. If 0.0, no maximum distance is applied and the
                number of amplitudes will be exactly n_amplitudes. Defaults to 0.0.
            reference_magnitude (float, optional): The reference magnitude to retrieve
                the amplitudes for. Defaults to 1.0.
            peak_amplitude (PeakAmplitude, optional): The type of peak amplitude to
                retrieve. Defaults to "absolute".

        Returns:
            ModelledAmplitude: The modelled amplitudes.

        Raises:
            ValueError: If there are not enough amplitudes in the specified range.
            ValueError: If the peak amplitude type is unknown.
        """
        magnitude_idx = np.where(self._magnitudes == reference_magnitude)[0]
        if not magnitude_idx.size:
            raise ValueError(f"No amplitudes for magnitude {reference_magnitude}.")

        site_distances = np.abs(self._distances[magnitude_idx] - distance)
        distance_idx = np.argsort(site_distances)

        idx = distance_idx[:n_amplitudes]

        distances = site_distances[idx]
        if distance_cutoff and distances.max() > distance_cutoff:
            raise ValueError(
                f"Not enough amplitudes at distance {distance}"
                f" at cutoff {distance_cutoff}"
            )

        match peak_amplitude:
            case "horizontal":
                amplitudes = self._horizontal[magnitude_idx][idx]
            case "vertical":
                amplitudes = self._vertical[magnitude_idx][idx]
            case "absolute":
                amplitudes = self._absolute[magnitude_idx][idx]
            case _:
                raise ValueError(f"Unknown peak amplitude type {peak_amplitude}.")

        median = float(np.median(amplitudes))
        return ModelledAmplitude(
            magnitude=reference_magnitude,
            peak_amplitude=peak_amplitude,
            quantity=self.quantity,
            distance_epi=distance,
            average=amplitudes.mean(),
            std=amplitudes.std(),
            median=median,
            mad=float(np.median(np.abs(amplitudes - median))),
        )

    def fill(
        self,
        receivers: list[gf.Receiver],
        traces: list[list[Trace]],
        magnitudes: list[float],
    ) -> None:
        for receiver, rcv_traces, magnitude in zip(
            receivers, traces, magnitudes, strict=True
        ):
            self.site_amplitudes.append(
                SiteAmplitude.from_traces(receiver, rcv_traces, magnitude)
            )
        self._clear_cache()

    def distance_range(self) -> Range:
        """Get the distance range of the site amplitudes.

        Returns:
            Range: The distance range.
        """
        return Range(self._distances.min(), self._distances.max())

    @property
    def n_amplitudes(self) -> int:
        """Get the number of amplitudes in the collection.

        Returns:
            int: The number of amplitudes.
        """
        return len(self.site_amplitudes)

    def plot(
        self,
        axes: Axes | None = None,
        reference_magnitude: float = 1.0,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> None:
        from matplotlib.ticker import FuncFormatter

        if axes is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        else:
            ax = axes

        labels: dict[MeasurementUnit, str] = {
            "displacement": "u [nm]",
            "velocity": "v [nm/s]",
            "acceleration": "a [nm/s²]",
        }
        interp_amplitudes: list[ModelledAmplitude] = []
        for distance in np.arange(*self.distance_range(), 250.0):
            interp_amplitudes.append(
                self.get_amplitude_model(
                    distance=distance,
                    n_amplitudes=50,
                    peak_amplitude=peak_amplitude,
                    reference_magnitude=reference_magnitude,
                )
            )

        interp_dists = np.array([amp.distance_epi for amp in interp_amplitudes])
        interp_amps = np.array([amp.median for amp in interp_amplitudes])
        # interp_std = np.array([amp.std for amp in interp_amplitudes])
        interp_mad = np.array([amp.mad for amp in interp_amplitudes])

        site_amplitudes = getattr(self, f"_{peak_amplitude.replace('peak_', '')}")
        dynamic = Range.from_list(site_amplitudes)

        ax.scatter(
            self._distances,
            site_amplitudes / NM,
            marker="o",
            c="k",
            s=2.0,
            alpha=0.05,
        )
        ax.scatter(
            interp_dists,
            interp_amps / NM,
            marker="o",
            c="forestgreen",
            s=6.0,
            alpha=1.0,
        )
        ax.fill_between(
            interp_dists,
            (interp_amps - interp_mad) / NM,
            (interp_amps + interp_mad) / NM,
            alpha=0.1,
            color="forestgreen",
        )

        ax.set_xlabel("Epicentral Distance [km]")
        ax.set_ylabel(labels[self.quantity])
        ax.set_yscale("log")
        ax.text(
            0.025,
            0.025,
            f"""n={self.n_amplitudes}
$M_w^{{ref}}$={reference_magnitude}
$z$={self.source_depth / KM} km
$v_r$=[{self.rupture_velocities.min}, {self.rupture_velocities.max}]$\\cdot v_s$
$\\Delta\\sigma$=[{self.stress_drop.min / 1e6}, {self.stress_drop.max / 1e6}] MPa
$f$=[{self.frequency_range.min}, {self.frequency_range.max}] Hz""",
            alpha=0.5,
            transform=ax.transAxes,
            va="bottom",
            fontsize="small",
        )
        ax.text(
            0.95,
            0.95,
            f"Measure: {peak_amplitude}\n"
            f"Dynamic: {(dynamic.max - dynamic.min) / NM:g}",
            alpha=0.5,
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.grid(alpha=0.1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: x / KM))
        if axes is None:
            plt.show()


class PeakAmplitudesStore(PeakAmplitudesBase):
    uuid: UUID = Field(
        default_factory=uuid4,
        description="Unique ID of the amplitude store.",
    )
    amplitude_collections: list[SiteAmplitudesCollection] = Field(
        default_factory=list,
        description="Site amplitudes per source depth.",
    )
    frequency_range: Range = Field(
        ...,
        description="Frequency range for the peak amplitude.",
    )
    gf_store_hash: str = Field(
        default="",
        description="Hash of the GF store configuration.",
    )
    magnitude_range: Range = Field(
        default=Range(0.0, 6.0),
        description="Range of moment magnitudes for the seismic sources.",
    )

    _rng: np.random.Generator = PrivateAttr(default_factory=np.random.default_rng)
    _access_locks: dict[int, asyncio.Lock] = PrivateAttr(
        default_factory=lambda: defaultdict(asyncio.Lock)
    )
    _engine: ClassVar[gf.LocalEngine | None] = None
    _cache_dir: ClassVar[Path | None] = None

    model_config: ConfigDict = {"extra": "ignore"}

    @classmethod
    def set_engine(cls, engine: gf.LocalEngine) -> None:
        """Set the GF engine for the store.

        Args:
            engine (gf.LocalEngine): The engine to use.
        """
        cls._engine = engine

    @classmethod
    def set_cache_dir(cls, cache_dir: Path) -> None:
        """Set the cache directory for the store.

        Args:
            cache_dir (Path): The cache directory to use.
        """
        cls._cache_dir = cache_dir

    @classmethod
    def from_selector(cls, selector: PeakAmplitudesBase) -> Self:
        """Create a new PeakAmplitudesStore from the given selector.

        Args:
            selector (PeakAmplitudesSelector): The selector to use.

        Returns:
            PeakAmplitudesStore: The newly created store.
        """
        if cls._engine is None:
            raise EnvironmentError(
                "No GF engine available to determine frequency range."
            )
        store: gf.Store = cls._engine.get_store(selector.gf_store_id)
        config = store.config
        if not isinstance(config, gf.ConfigTypeA):
            raise EnvironmentError("GF store is not of type ConfigTypeA.")

        store_frequency_range = Range(0.0, 1.0 / config.deltat)
        if (
            selector.frequency_range
            and selector.frequency_range.max > store_frequency_range.max
        ):
            raise ValueError(
                f"Selector frequency range {selector.frequency_range} "
                f"exceeds store frequency range {store_frequency_range}."
            )

        kwargs = selector.model_dump()
        kwargs["frequency_range"] = selector.frequency_range or store_frequency_range
        kwargs["gf_store_hash"] = store.create_store_hash()
        return cls(**kwargs)

    @property
    def source_depth_range(self) -> Range:
        return Range.from_list([sa.source_depth for sa in self.amplitude_collections])

    @property
    def gf_store_depth_range(self) -> Range:
        """Get the depth range of the GF store.

        Returns:
            Range: The depth range.
        """
        store = self.get_store()
        return Range(store.config.source_depth_min, store.config.source_depth_max)

    @property
    def gf_store_distance_range(self) -> Range:
        """Returns the distance range for the ground motion store.

        The distance range is determined by the minimum and maximum distances
        specified in the store's configuration. If the maximum distance exceeds
        the maximum distance allowed by the current object, it is truncated to
        match the maximum distance.

        Returns:
            Range: The distance range for the ground motion store.
        """
        store = self.get_store()
        return Range(
            min=store.config.distance_min,
            max=min(store.config.distance_max, self.max_distance),
        )

    def get_lock(self, source_depth: float, reference_magnitude: float) -> asyncio.Lock:
        """Get the lock for the given source depth and reference magnitude.

        Args:
            source_depth (float): The source depth.
            reference_magnitude (float): The reference magnitude.

        Returns:
            asyncio.Lock: The lock for the given source depth and reference magnitude.
        """
        return self._access_locks[hash((source_depth, reference_magnitude))]

    def get_store(self) -> gf.Store:
        """Load the GF store for the given store ID."""
        if self._engine is None:
            raise EnvironmentError("No GF engine available.")

        try:
            store = self._engine.get_store(self.gf_store_id)
        except Exception as exc:
            raise EnvironmentError(
                f"Failed to load GF store {self.gf_store_id}."
            ) from exc

        config = store.config
        if not isinstance(config, gf.ConfigTypeA):
            raise EnvironmentError("GF store is not of type ConfigTypeA.")
        if 1.0 / config.deltat < self.frequency_range.max:
            raise ValueError(
                f"Pyrocko GF store frequency {1.0 / config.deltat} too low."
            )
        return store

    def _get_random_source(
        self,
        depth: float,
        reference_magnitude: float,
        stf: Type[gf.STF] | None = None,
    ) -> MTSourceCircularCrack:
        """Generates a random seismic source with the given depth.

        Args:
            depth (float): The depth of the seismic source.
            reference_magnitude (float): The reference moment magnitude.
            stf (Type[gf.STF], optional): The source time function to use.
                Defaults to None.

        Returns:
            gf.MTSource: A random moment tensor source.
        """
        rng = self._rng
        store = self.get_store()
        velocity_model: LayeredModel = store.config.earthmodel_1d
        vs = np.interp(depth, velocity_model.profile("z"), velocity_model.profile("vs"))

        stress_drop = rng.uniform(*self.stress_drop)
        rupture_velocity = rng.uniform(*self.rupture_velocities) * vs

        radius = (
            pmt.magnitude_to_moment(reference_magnitude) * (7 / 16) / stress_drop
        ) ** (1 / 3)
        duration = 1.5 * radius / rupture_velocity
        moment_tensor = pmt.MomentTensor.random_dc(magnitude=reference_magnitude)
        return MTSourceCircularCrack(
            magnitude=reference_magnitude,
            stress_drop=stress_drop,
            radius=radius,
            m6=moment_tensor.m6(),
            depth=depth,
            duration=duration,
            stf=stf(effective_duration=duration) if stf else None,
        )

    def _get_random_targets(
        self,
        distance_range: Range,
        n_receivers: int,
    ) -> list[gf.Target]:
        """Generate a list of receivers with random angles and distances.

        Args:
            distance_range (Range): The range of distances to generate the
                receivers for.
            n_receivers (int): The number of receivers to generate.

        Returns:
            list[gf.Receiver]: A list of receivers with random angles and distances.
        """
        rng = self._rng
        _distance_range = np.array(distance_range)
        _distance_range[_distance_range <= 0.0] = 1.0  # Add an epsilon

        angles = rng.uniform(0.0, 360.0, size=n_receivers)
        distances = np.exp(rng.uniform(*np.log(_distance_range), size=n_receivers))
        targets: list[gf.Receiver] = []

        for i_receiver, (angle, distance) in enumerate(
            zip(angles, distances, strict=True)
        ):
            for component in get_args(Components):
                target = gf.Target(
                    quantity=self.quantity,
                    store_id=self.gf_store_id,
                    interpolation=self.gf_interpolation,
                    depth=0.0,
                    dip=_DIPS[component],
                    azimuth=angle + _AZIMUTHS[component],
                    north_shift=distance * np.cos(np.radians(angle)),
                    east_shift=distance * np.sin(np.radians(angle)),
                    codes=("PA", f"{i_receiver:05d}", "", component),
                )
                targets.append(target)
        return targets  # type: ignore

    async def compute_site_amplitudes(
        self,
        source_depth: float,
        reference_magnitude: float,
        n_sources: int = 200,
        n_targets_per_source: int = 20,
    ) -> SiteAmplitudesCollection:
        """Fills the moment magnitude store.

        Calculates the amplitudes for a given source depth and reference magnitude.

        Args:
            source_depth (float): The depth of the seismic source.
            reference_magnitude (float): The reference moment magnitude.
            n_targets_per_source (int, optional): The number of target locations to
                calculate amplitudes for. Defaults to 20.
            n_sources (int, optional): The number of source locations to generate
                random sources from. Defaults to 100.
        """
        if self._engine is None:
            raise EnvironmentError("No GF engine available.")
        if not self.gf_store_depth_range.inside(source_depth):
            raise ValueError(f"Source depth {source_depth} outside GF store range.")

        engine = self._engine

        target_distances = self.gf_store_distance_range
        logger.info(
            "calculating %d %s amplitudes for Mw %.1f at depth %.1f",
            n_sources * n_targets_per_source,
            self.quantity,
            reference_magnitude,
            source_depth,
        )

        receivers = []
        receiver_traces = []
        magnitudes = []
        status = PROGRESS.add_task(
            f"Calculating Mw {reference_magnitude} amplitudes for depth {source_depth}",
            total=n_sources,
        )
        for _ in range(n_sources):
            targets = self._get_random_targets(target_distances, n_targets_per_source)
            source = self._get_random_source(source_depth, reference_magnitude)
            response = await asyncio.to_thread(engine.process, source, targets)

            traces: list[Trace] = response.pyrocko_traces()
            for tr in traces:
                tr.transfer(transfer_function=BruneResponse(duration=source.duration))
                if self.frequency_range:
                    if self.frequency_range.min > 0.0:
                        tr.highpass(4, self.frequency_range.min, demean=False)
                    if self.frequency_range.max < 1.0 / tr.deltat:
                        tr.lowpass(4, self.frequency_range.max, demean=False)

            for nsl, grp_traces in itertools.groupby(
                traces, key=lambda tr: tr.nslc_id[:3]
            ):
                receivers.append(_get_target(targets, nsl))
                receiver_traces.append(list(grp_traces))
                magnitudes.append(reference_magnitude)
            PROGRESS.update(status, advance=1)
        PROGRESS.remove_task(status)

        try:
            collection = self.get_collection(source_depth)
        except KeyError:
            collection = self.new_collection(source_depth)
        collection.fill(receivers, receiver_traces, magnitudes)
        self.save()
        return collection

    async def fill_source_depth_range(
        self,
        depth_min: float | None = None,
        depth_max: float | None = None,
        depth_delta: float | None = None,
        n_sources: int = 400,
        n_targets_per_source: int = 20,
    ) -> None:
        """Fills the source depth range with seismic data.

        Args:
            depth_min (float): The minimum depth of the source in meters.
                If None, it uses the extent from the GF store.
                Defaults to None.
            depth_max (float): The maximum depth of the source in meters.
                If None, it uses the extent from the GF store.
                Defaults to None.
            depth_delta (float | None, optional): The depth increment in meters.
                If None, it uses the default value from the GF store configuration.
                Defaults to None.
            n_sources (int, optional): The number of random source realisation
                per source depth. Defaults to 400.
            n_targets_per_source (int, optional): The number of targets per
                source depth. Targets are randomized in distance and azimuth.
                Defaults to 20.
        """
        store = self.get_store()

        gf_depth_delta = store.config.source_depth_delta
        gf_depth_min = store.config.source_depth_min
        gf_depth_max = store.config.source_depth_max

        depth_min = depth_min or gf_depth_min
        depth_max = depth_max or gf_depth_max
        depth_delta = depth_delta or gf_depth_delta

        depths = np.arange(gf_depth_min, gf_depth_max, depth_delta)
        calculate_depths = depths[(depths >= depth_min) & (depths <= depth_max)]

        stored_depths = [sa.source_depth for sa in self.amplitude_collections]
        logger.debug("filling source depths %s", calculate_depths)
        for depth in calculate_depths:
            if depth in stored_depths:
                if store.create_store_hash() != self.gf_store_hash:
                    self.remove_collection(depth)
                else:
                    continue
            async with self.get_lock(depth, self.reference_magnitude):
                await self.compute_site_amplitudes(
                    reference_magnitude=self.reference_magnitude,
                    source_depth=depth,
                    n_sources=n_sources,
                    n_targets_per_source=n_targets_per_source,
                )

    def get_collection(self, source_depth: float) -> SiteAmplitudesCollection:
        """Get the site amplitudes collection for the given source depth.

        Args:
            source_depth (float): The source depth.

        Returns:
            SiteAmplitudesCollection: The site amplitudes collection.
        """
        for site_amplitudes in self.amplitude_collections:
            if site_amplitudes.source_depth == source_depth:
                return site_amplitudes
        raise KeyError(f"No site amplitudes for depth {source_depth}.")

    def new_collection(self, source_depth: float) -> SiteAmplitudesCollection:
        """Creates a new SiteAmplitudesCollection object.

        For the given depth and add it to the list of site amplitudes.

        Args:
            source_depth (float): The depth for which the site amplitudes collection is
                created.

        Returns:
            SiteAmplitudesCollection: The newly created SiteAmplitudesCollection object.
        """
        logger.debug("creating new site amplitudes for depth %f", source_depth)
        self.remove_collection(source_depth)
        collection = SiteAmplitudesCollection(
            source_depth=source_depth,
            **self.model_dump(exclude={"site_amplitudes"}),
        )
        self.amplitude_collections.append(collection)
        return collection

    def remove_collection(self, depth: float) -> None:
        """Removes the site amplitudes collection for the given depth.

        Args:
            depth (float): The depth for which the site amplitudes collection is
                removed.
        """
        try:
            collection = self.get_collection(depth)
            self.amplitude_collections.remove(collection)
            logger.debug("removed site amplitudes for depth %f", depth)
        except KeyError:
            pass

    async def get_amplitude_model(
        self,
        source_depth: float,
        distance: float,
        n_amplitudes: int = 25,
        distance_cutoff: float = 0.0,
        reference_magnitude: float | None = None,
        peak_amplitude: PeakAmplitude = "absolute",
        auto_fill: bool = True,
        interpolation: Literal["nearest", "linear"] = "linear",
    ) -> ModelledAmplitude:
        """Retrieves the amplitude for a given depth and distance.

        Args:
            source_depth (float): The depth of the event.
            distance (float): The epicentral distance from the event.
            n_amplitudes (int, optional): The number of amplitudes to retrieve.
                Defaults to 10.
            distance_cutoff (float, optional): The maximum distance allowed for
                the retrieved amplitudes. If 0.0, no maximum distance is applied and the
                number of amplitudes will be exactly n_amplitudes. Defaults to 0.0.
            reference_magnitude (float, optional): The reference moment magnitude
                for the amplitudes. Defaults to 1.0.
            peak_amplitude (PeakAmplitude, optional): The type of peak amplitude to
                retrieve. Defaults to "absolute".
            auto_fill (bool, optional): If True, the site amplitudes for
                depth-reference magnitude combinations are calculated
                if they are not available. Defaults to True.
            interpolation (Literal["nearest", "linear"], optional): The depth
                interpolation method to use. Defaults to "linear".

        Returns:
            ModelledAmplitude: The modelled amplitude for the given depth, distance and
                reference magnitude.
        """
        if not self.source_depth_range.inside(source_depth):
            raise ValueError(f"Source depth {source_depth} outside range.")

        source_depths = np.array([sa.source_depth for sa in self.amplitude_collections])
        reference_magnitude = (
            self.reference_magnitude
            if reference_magnitude is None
            else reference_magnitude
        )

        match interpolation:
            case "nearest":
                idx = [np.abs(source_depths - source_depth).argmin()]
            case "linear":
                idx = np.argsort(np.abs(source_depths - source_depth))[:2]
            case _:
                raise ValueError(f"Unknown interpolation method {interpolation}.")

        collections = [self.amplitude_collections[i] for i in idx]

        amplitudes: list[ModelledAmplitude] = []

        for collection in collections:
            lock = self.get_lock(collection.source_depth, reference_magnitude)
            try:
                await lock.acquire()
                amplitude = collection.get_amplitude_model(
                    distance=distance,
                    n_amplitudes=n_amplitudes,
                    distance_cutoff=distance_cutoff,
                    peak_amplitude=peak_amplitude,
                    reference_magnitude=reference_magnitude,
                )
                amplitudes.append(amplitude)
            except ValueError as e:
                logger.exception(e)
                if auto_fill:
                    await self.compute_site_amplitudes(
                        source_depth=collection.source_depth,
                        reference_magnitude=reference_magnitude,
                    )
                    lock.release()
                    return await self.get_amplitude_model(
                        source_depth=source_depth,
                        distance=distance,
                        n_amplitudes=n_amplitudes,
                        reference_magnitude=reference_magnitude,
                        distance_cutoff=distance_cutoff,
                        peak_amplitude=peak_amplitude,
                        interpolation=interpolation,
                        auto_fill=True,
                    )
                lock.release()
                raise
            lock.release()

        if not amplitudes:
            raise ValueError(f"No site amplitudes for depth {source_depth}.")

        match interpolation:
            case "nearest":
                amplitude = amplitudes[0]

            case "linear":
                if len(amplitudes) == 1:
                    return amplitudes[0]
                depths = source_depths[idx]
                weight = abs((source_depth - depths[0]) / abs(depths[1] - depths[0]))
                amplitude = amplitudes[0].combine(amplitudes[1], weight=weight)
            case _:
                raise ValueError(f"Unknown interpolation method {interpolation}.")
        if amplitude.median == 0.0:
            raise ValueError(f"Median amplitude is zero for depth {source_depth}.")
        return amplitude

    async def find_moment_magnitude(
        self,
        source_depth: float,
        distance: float,
        observed_amplitude: float,
        n_amplitudes: int = 25,
        distance_cutoff: float = 0.0,
        initial_reference_magnitude: float = 1.0,
        peak_amplitude: PeakAmplitude = "absolute",
        interpolation: Literal["nearest", "linear"] = "linear",
    ) -> tuple[float, ModelledAmplitude]:
        """Get the moment magnitude for the given observed amplitude.

        Args:
            source_depth (float): The depth of the event.
            distance (float): The epicentral distance from the event.
            observed_amplitude (float): The observed amplitude.
            n_amplitudes (int, optional): The number of amplitudes to retrieve.
                Defaults to 10.
            initial_reference_magnitude (float, optional): The initial reference
                moment magnitude to use. Defaults to 1.0.
            distance_cutoff (float, optional): The maximum distance allowed for
                the retrieved amplitudes. If 0.0, no maximum distance is applied and the
                number of amplitudes will be exactly n_amplitudes. Defaults to 0.0.
            peak_amplitude (PeakAmplitude, optional): The type of peak amplitude to
                retrieve. Defaults to "absolute".
            interpolation (Literal["nearest", "linear"], optional): The depth
                interpolation method to use. Defaults to "linear".

        Returns:
            float: The moment magnitude.
        """
        cache: list[tuple[float, float, ModelledAmplitude]] = []

        def get_cache(reference_magnitude: float) -> tuple[float, ModelledAmplitude]:
            for mag, est, model in cache:
                if mag == reference_magnitude:
                    return est, model
            raise KeyError(f"No estimate for magnitude {reference_magnitude}.")

        async def estimate_magnitude(
            reference_magnitude: float,
        ) -> tuple[float, ModelledAmplitude]:
            try:
                return get_cache(reference_magnitude)
            except KeyError:
                model = await self.get_amplitude_model(
                    reference_magnitude=reference_magnitude,
                    source_depth=source_depth,
                    distance=distance,
                    n_amplitudes=n_amplitudes,
                    distance_cutoff=distance_cutoff,
                    peak_amplitude=peak_amplitude,
                    interpolation=interpolation,
                )
                est_magnitude = model.estimate_magnitude(observed_amplitude)
                cache.append((reference_magnitude, est_magnitude, model))
                return est_magnitude, model

        reference_mag = initial_reference_magnitude
        for _ in range(3):
            est_magnitude, _ = await estimate_magnitude(reference_mag)
            rounded_mag = np.round(est_magnitude, 0)
            explore_mags = np.array([rounded_mag - 1, rounded_mag, rounded_mag + 1])

            predictions = [await estimate_magnitude(mag) for mag in explore_mags]
            predicted_mags = np.array([mag for mag, _ in predictions])
            models = [model for _, model in predictions]

            magnitude_differences = np.abs(predicted_mags - explore_mags)
            min_diff = np.argmin(magnitude_differences)

            if min_diff == 1:
                return predicted_mags[1], models[1]
            reference_mag = explore_mags[min_diff]
        return predicted_mags[min_diff], models[min_diff]

    def hash(self) -> str:
        """Calculate the hash of the store from store parameters.

        Returns:
            str: The hash of the store.
        """
        data = struct.pack(
            "ddddddddsss",
            self.reference_magnitude,
            self.max_distance,
            *self.frequency_range,
            *self.rupture_velocities,
            *self.stress_drop,
            self.gf_store_id.encode(),
            self.gf_interpolation.encode(),
            self.gf_store_hash.encode(),
        )
        return hashlib.sha1(data).hexdigest()

    def is_suited(self, selector: PeakAmplitudesBase) -> bool:
        """Check if the given selector is suited for this store.

        Args:
            selector (PeakAmpliutdesSelector): The selector to check.

        Returns:
            bool: True if the selector is suited for this store.
        """
        result = (
            self.gf_store_id == selector.gf_store_id
            and self.gf_interpolation == selector.gf_interpolation
            and self.quantity == selector.quantity
            and self.reference_magnitude == selector.reference_magnitude
            and self.rupture_velocities == selector.rupture_velocities
            and self.stress_drop == selector.stress_drop
            and self.max_distance >= selector.max_distance
        )
        if selector.frequency_range:
            result = result and self.frequency_range == selector.frequency_range
        return result

    def __hash__(self) -> int:
        return hash(self.hash())

    def save(self, path: Path | None = None) -> None:
        """Save the site amplitudes to a JSON file.

        The site amplitudes are saved in a directory called 'site_amplitudes'
        within the cache directory. The file name is generated based on the store ID and
        a hash of the store parameters.
        """
        if not path:
            if not self._cache_dir:
                return
            path = self._cache_dir

        file = path / f"{self.gf_store_id}-{self.quantity}-{self.hash()}.json"
        logger.info("saving site amplitudes to %s", file)
        file.write_text(self.model_dump_json())


class CacheStats(NamedTuple):
    path: Path
    n_stores: int
    bytes: ByteSize


class PeakAmplitudeStoreCache:
    cache_dir: Path
    engine: gf.LocalEngine

    def __init__(self, cache_dir: Path, engine: gf.LocalEngine | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("cache size: %s", human_readable_bytes(self.cache_stats().bytes))
        self.clean_cache()

        self.engine = engine or gf.LocalEngine(store_superdirs=["."])
        PeakAmplitudesStore.set_engine(engine)
        PeakAmplitudesStore.set_cache_dir(cache_dir)

    def clear_cache(self):
        """Clear the cache directory.

        This method deletes all files in the cache directory.
        """
        logger.info("clearing cache directory %s", self.cache_dir)
        for file in self.cache_dir.glob("*"):
            file.unlink()

    def clean_cache(self, keep_files: int = 100) -> None:
        """Clean the cache directory.

        Args:
            keep_files (int, optional): The number of most recent files to keep in the
                cache directory. Defaults to 100.
        """
        files = sorted(self.cache_dir.glob("*"), key=lambda f: f.stat().st_mtime)
        if len(files) <= keep_files:
            return
        logger.info("cleaning cache directory %s", self.cache_dir)
        for file in files[keep_files:]:
            file.unlink()

    def cache_stats(self) -> CacheStats:
        """Get the cache statistics.

        Returns:
            CacheStats: The cache statistics.
        """
        n_stores = 0
        nbytes = 0
        for file in self.cache_dir.glob("*.json"):
            n_stores += 1
            nbytes += file.stat().st_size
        return CacheStats(path=self.cache_dir, n_stores=n_stores, bytes=nbytes)

    def get_cached_stores(
        self, store_id: str, quantity: MeasurementUnit
    ) -> list[PeakAmplitudesStore]:
        """Get the cached peak amplitude stores for the given store ID and quantity.

        Args:
            store_id (str): The store ID.
            quantity (MeasurementUnit): The quantity.

        Returns:
            list[PeakAmplitudesStore]: A list of peak amplitude stores.
        """
        stores = []
        for file in self.cache_dir.glob("*.json"):
            try:
                store_id, quantity, _ = file.stem.split("-")  # type: ignore
            except ValueError:
                logger.warning("Invalid file name %s, deleting file", file)
                file.unlink()

            if store_id == store_id and quantity == quantity:
                try:
                    store = PeakAmplitudesStore.model_validate_json(file.read_text())
                except ValidationError:
                    logger.warning("Invalid store %s, deleting file", file)
                    file.unlink()
                    continue
                stores.append(store)
        return stores

    def get_store(self, selector: PeakAmplitudesBase) -> PeakAmplitudesStore:
        """Get a peak amplitude store for the given selector.

        Either from the cache or by creating a new store.

        Args:
            selector (PeakAmplitudesSelector): The selector to use.

        Returns:
            PeakAmplitudesStore: The peak amplitude store.
        """
        for store in self.get_cached_stores(selector.gf_store_id, selector.quantity):
            if store.is_suited(selector):
                return store
        return PeakAmplitudesStore.from_selector(selector)

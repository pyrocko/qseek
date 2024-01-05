from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, NamedTuple
from uuid import UUID, uuid4

import numpy as np
import pyrocko.moment_tensor as pmt
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationError,
)
from pyrocko import gf
from rich.progress import track
from typing_extensions import Self

from qseek.utils import (
    ChannelSelector,
    ChannelSelectors,
    MeasurementUnit,
    Range,
)

if TYPE_CHECKING:
    from pyrocko.trace import Trace

KM = 1e3

logger = logging.getLogger(__name__)

PeakAmplitude = Literal["horizontal", "vertical", "absolute"]
Interpolation = Literal["nearest_neighbour", "multilinear"]


def trace_amplitude(traces: list[Trace], channel_selector: ChannelSelector) -> float:
    """
    Normalize traces channels.

    Args:
        traces (list[Trace]): A list of traces to normalize.
        components (str): The components to normalize.

    Returns:
        Trace: The normalized trace.

    Raises:
        KeyError: If there are no traces to normalize.
    """
    trace_selection = channel_selector(traces)
    if not trace_selection:
        raise KeyError("No traces to normalize.")

    if len(trace_selection) == 1:
        tr = trace_selection[0].copy()
        data = np.abs(tr.ydata)
    else:
        data = np.array([tr.ydata for tr in trace_selection])
        data = np.linalg.norm(data, axis=0)
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

    reference_magnitude: float = Field(
        default=1.0,
        ge=-1.0,
        le=8.0,
        description="Reference magnitude in Mw.",
    )
    rupture_velocities: Range = Field(
        default=(0.9, 1.0),
        description="Rupture velocity range as fraction of the shear wave velocity.",
    )
    stress_drop: Range = Field(
        default=(1.0e6, 10.0e6),
        description="Stress drop range in MPa.",
    )
    gf_interpolation: Interpolation = Field(
        default="nearest_neighbour",
        description="Interpolation method for the Pyrocko GF Store",
    )


class SiteAmplitude(NamedTuple):
    distance: float
    peak_horizontal: float
    peak_vertical: float
    peak_absolute: float

    @classmethod
    def from_traces(cls, receiver: gf.Receiver, traces: list[Trace]) -> Self:
        surface_distance = np.sqrt(receiver.north_shift**2 + receiver.east_shift**2)
        return cls(
            distance=surface_distance,
            peak_horizontal=trace_amplitude(
                traces,
                ChannelSelectors.Horizontal,
            ),
            peak_vertical=trace_amplitude(
                traces,
                ChannelSelectors.Vertical,
            ),
            peak_absolute=trace_amplitude(
                traces,
                ChannelSelectors.All,
            ),
        )


class ModelledAmplitude(NamedTuple):
    distance: float
    peak_amplitude: PeakAmplitude
    amplitude_avg: float
    amplitude_median: float
    std: float


class SiteAmplitudesCollection(BaseModel):
    source_depth: float
    site_amplitudes: list[SiteAmplitude] = Field(default_factory=list)

    @staticmethod
    def _get_numpy_attribute(attribute: str) -> Callable:
        def wrapped(self) -> np.ndarray:
            return np.array([getattr(sa, attribute) for sa in self.site_amplitudes])

        return wrapped

    _distances = cached_property(_get_numpy_attribute("distance"))
    _vertical = cached_property(_get_numpy_attribute("peak_vertical"))
    _absolute = cached_property(_get_numpy_attribute("peak_absolute"))
    _horizontal = cached_property(_get_numpy_attribute("peak_horizontal"))

    def get_amplitude(
        self,
        distance: float,
        n_amplitudes: int,
        max_distance: float,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> ModelledAmplitude:
        """
        Get the amplitudes for a given distance.

        Args:
            distance (float): The distance at which to retrieve the amplitudes.
            n_amplitudes (int): The number of amplitudes to retrieve.
            max_distance (float): The maximum distance allowed for
                the retrieved amplitudes.
            peak_amplitude (PeakAmplitude, optional): The type of peak amplitude to
                retrieve. Defaults to "absolute".

        Returns:
            ModelledAmplitude: The modelled amplitudes.

        Raises:
            ValueError: If there are not enough amplitudes in the specified range.
            ValueError: If the peak amplitude type is unknown.
        """
        distance_idx = np.argsort(np.abs(self._distances - distance))
        idx = distance_idx[:n_amplitudes]
        distances = self._distances[idx]
        if distances.max() > max_distance:
            raise ValueError(f"Not enough amplitudes in range {max_distance}.")

        match peak_amplitude:
            case "horizontal":
                amplitudes = self._horizontal[idx]
            case "vertical":
                amplitudes = self._vertical[idx]
            case "absolute":
                amplitudes = self._absolute[idx]
            case _:
                raise ValueError(f"Unknown peak amplitude type {peak_amplitude}.")

        return ModelledAmplitude(
            distance=distance,
            peak_amplitude=peak_amplitude,
            amplitude_avg=amplitudes.mean(),
            amplitude_median=amplitudes.mean(),
            std=amplitudes.std(),
        )

    def fill(self, receivers: list[gf.Receiver], traces: list[list[Trace]]) -> None:
        for receiver, rcv_traces in zip(receivers, traces, strict=True):
            self.site_amplitudes.append(SiteAmplitude.from_traces(receiver, rcv_traces))
        self.clear()

    def clear(self) -> None:
        del self._distances
        del self._horizontal
        del self._vertical
        del self._absolute


class PeakAmplitudesStore(PeakAmplitudesBase):
    uuid: UUID = Field(
        default_factory=uuid4,
        description="Unique ID of the amplitude store.",
    )
    site_amplitudes: list[SiteAmplitudesCollection] = Field(
        default_factory=list,
        description="Site amplitudes per source depth.",
    )
    frequency_range: Range = Field(
        ...,
        description="Frequency range for the peak amplitude.",
    )

    _rng: np.random.Generator = PrivateAttr(default_factory=np.random.default_rng)
    _engine: ClassVar[gf.LocalEngine | None] = None
    _cache_dir: ClassVar[Path | None] = None

    @classmethod
    def set_engine(cls, engine: gf.LocalEngine) -> None:
        """
        Set the GF engine for the store.

        Args:
            engine (gf.LocalEngine): The engine to use.
        """
        cls._engine = engine

    @classmethod
    def from_selector(cls, selector: PeakAmplitudesBase) -> Self:
        """
        Create a new PeakAmplitudesStore from the given selector.

        Args:
            selector (PeakAmplitudesSelector): The selector to use.

        Returns:
            PeakAmplitudesStore: The newly created store.
        """

        if cls._engine is None:
            raise EnvironmentError(
                "No GF engine available to determine frequency range."
            )
        config = cls._engine.get_store(selector.gf_store_id).config
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

        return cls(
            gf_store_id=selector.gf_store_id,
            quantity=selector.quantity,
            reference_magnitude=selector.reference_magnitude,
            rupture_velocities=selector.rupture_velocities,
            stress_drop=selector.stress_drop,
            frequency_range=selector.frequency_range or store_frequency_range,
        )

    def get_store(self) -> gf.Store:
        """
        Load the GF store for the given store ID.
        """
        if self._engine is None:
            raise EnvironmentError("No GF engine available.")

        try:
            store = self._engine.get_store(self.gf_store_id)
        except Exception as exc:
            raise EnvironmentError(
                f"Failed to load GF store {self.gf_store_id}."
            ) from exc

        meta = store.meta
        if not isinstance(meta, gf.ConfigTypeA):
            raise EnvironmentError("GF store is not of type ConfigTypeA.")

        if 1.0 / meta.deltat < self.frequency_range.max:
            raise ValueError(f"GF store maximum frequency {1.0 / meta.deltat} too low.")

        return store

    def is_suited(self, selector: PeakAmplitudesBase) -> bool:
        """
        Check if the given selector is suited for this store.

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
            and self.rupture_velocities.min <= selector.rupture_velocities.min
            and self.rupture_velocities.max >= selector.rupture_velocities.max
            and self.stress_drop.min <= selector.stress_drop.min
            and self.stress_drop.max >= selector.stress_drop.max
        )
        if selector.frequency_range:
            result = result and self.frequency_range.max <= selector.frequency_range.max
        return result

    def _get_random_source(self, depth: float) -> gf.MTSource:
        """
        Generates a random seismic source with the given depth.

        Args:
            depth (float): The depth of the seismic source.

        Returns:
            gf.MTSource: A random moment tensor source.
        """
        rng = self._rng
        stress_drop = rng.uniform(*self.stress_drop)
        rupture_velocity = rng.uniform(*self.rupture_velocities)

        radius = (
            pmt.magnitude_to_moment(self.reference_magnitude) * 7.0 / 16.0 / stress_drop
        ) ** (1.0 / 3.0)
        duration = 1.5 * radius / rupture_velocity
        moment_tensor = pmt.MomentTensor.random_dc(magnitude=self.reference_magnitude)
        return gf.MTSource(
            m6=moment_tensor.m6(),
            depth=depth,
            std=gf.HalfSinusoidSTF(effective_duration=duration),
        )

    def _get_targets(
        self,
        distance_range: Range,
        n_receivers: int,
    ) -> list[gf.Target]:
        """
        Generate a list of receivers with random angles and distances.

        Args:
            n_receivers (int): The number of receivers to generate.

        Returns:
            list[gf.Receiver]: A list of receivers with random angles and distances.
        """
        rng = self._rng
        angles = rng.uniform(0.0, 360.0, size=n_receivers)
        distances = np.exp(rng.uniform(*np.log(distance_range), size=n_receivers))
        targets: list[gf.Receiver] = []

        for i_receiver, (angle, distance) in enumerate(
            zip(angles, distances, strict=True)
        ):
            for component in "ZRT":
                target = gf.Target(
                    quantity=self.quantity,
                    store_id=self.gf_store_id,
                    interpolation="nearest",
                    depth=0.0,
                    north_shift=distance * np.cos(np.radians(angle)),
                    east_shift=distance * np.sin(np.radians(angle)),
                    codes=("", f"{i_receiver:04d}", component),
                )
                targets.append(target)
        return targets  # type: ignore

    async def _calculate_amplitudes(
        self,
        source_depth: float,
        n_targets: int = 20,
        n_sources: int = 100,
    ) -> None:
        if self._engine is None:
            raise EnvironmentError("No GF engine available.")

        store = self.get_store()

        engine = self._engine
        depths = Range(store.config.distance_min, store.config.distance_max)

        async def get_modelled_waveforms() -> gf.Response:
            targets = self._get_targets(depths, n_targets)
            source = self._get_random_source(source_depth)
            return await asyncio.to_thread(engine.process(source, targets))

        receivers = []
        receiver_traces = []
        logger.info(
            "calculating %d amplitudes for depth %f",
            n_sources * n_targets,
            source_depth,
        )
        try:
            collection = self.get_collection(source_depth)
        except KeyError:
            collection = self.new_collection(source_depth)

        for _ in track(
            range(n_sources),
            total=n_sources,
            description="Calculating amplitudes",
        ):
            response = await get_modelled_waveforms()
            for _, target, traces in response.iter_results():
                for tr in traces:
                    if self.frequency_range:
                        tr.highpass(4, self.frequency_range.min, demean=False)
                        tr.lowpass(4, self.frequency_range.max, demean=False)

                receivers.append(target)
                receiver_traces.append(traces)

        collection.fill(receivers, receiver_traces)
        self.save()

    def get_collection(self, depth: float) -> SiteAmplitudesCollection:
        """
        Get the site amplitudes collection for the given source depth.

        Args:
            depth (float): The source depth.

        Returns:
            SiteAmplitudesCollection: The site amplitudes collection.
        """
        for site_amplitudes in self.site_amplitudes:
            if site_amplitudes.source_depth == depth:
                return site_amplitudes
        raise KeyError(f"No site amplitudes for depth {depth}.")

    def new_collection(self, depth: float) -> SiteAmplitudesCollection:
        """
        Creates a new SiteAmplitudesCollection object for the given depth and
        adds it to the list of site amplitudes.

        Args:
            depth (float): The depth for which the site amplitudes collection is created.

        Returns:
            SiteAmplitudesCollection: The newly created SiteAmplitudesCollection object.
        """
        if (collection := self.get_collection(depth)) is not None:
            self.site_amplitudes.remove(collection)
        collection = SiteAmplitudesCollection(source_depth=depth)
        self.site_amplitudes.append(collection)
        return collection

    def get_amplitude(
        self,
        depth: float,
        distance: float,
        n_amplitudes: int = 10,
        max_distance: float = 1.0 * KM,
        peak_amplitude: PeakAmplitude = "absolute",
        auto_fill: bool = True,
    ) -> ModelledAmplitude:
        """
        Retrieves the amplitude for a given depth and distance.

        Args:
            depth (float): The depth of the event.
            distance (float): The distance from the event.
            n_amplitudes (int, optional): The number of amplitudes to retrieve.
                Defaults to 10.
            max_distance (float, optional): The maximum distance to consider in [m].
                Defaults to 1000.0.
            peak_amplitude (PeakAmplitude, optional): The type of peak amplitude to
                retrieve. Defaults to "absolute".
            auto_fill (bool, optional): If True, the site amplitudes are calculated
                if they are not available. Defaults to True.

        Returns:
            ModelledAmplitude: The modelled amplitude for the given depth and distance.
        """
        collection = self.get_collection(depth)
        try:
            return collection.get_amplitude(
                distance=distance,
                n_amplitudes=n_amplitudes,
                max_distance=max_distance,
                peak_amplitude=peak_amplitude,
            )
        except ValueError:
            if auto_fill:
                asyncio.run(self._calculate_amplitudes(depth))
                logger.info("auto-filling site amplitudes for depth %f", depth)
                return self.get_amplitude(
                    depth=depth,
                    distance=distance,
                    n_amplitudes=n_amplitudes,
                    max_distance=max_distance,
                    peak_amplitude=peak_amplitude,
                    auto_fill=True,
                )
            raise

    def hash(self) -> str:
        """
        Calculate the hash of the store from store parameters.

        Returns:
            str: The hash of the store.
        """
        data = struct.pack(
            "dddddddss",
            self.frequency_range.min,
            self.frequency_range.max,
            self.reference_magnitude,
            self.rupture_velocities.min,
            self.rupture_velocities.max,
            self.stress_drop.min,
            self.stress_drop.max,
            self.gf_store_id,
            self.gf_interpolation,
        )
        return hashlib.sha1(data).hexdigest()

    def __hash__(self) -> int:
        return hash(self.hash())

    def save(self, path: Path | None = None) -> None:
        """
        Save the site amplitudes to a JSON file.

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


class PeakAmplitudeStoreCache:
    cache_dir: Path
    engine: gf.LocalEngine

    def __init__(self, cache_dir: Path, engine: gf.LocalEngine | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.engine = engine or gf.LocalEngine(store_superdirs=["."])
        PeakAmplitudesStore.set_engine(engine)

    def clear_cache(self):
        """
        Clear the cache directory.
        """
        cache_dir = self.cache_dir / "site_amplitudes"
        for file in cache_dir.glob("*"):
            file.unlink()

    def get_cached_stores(
        self, store_id: str, quantity: MeasurementUnit
    ) -> list[PeakAmplitudesStore]:
        """
        Get the cached peak amplitude stores for the given store ID and quantity.

        Args:
            store_id (str): The store ID.
            quantity (MeasurementUnit): The quantity.

        Returns:
            list[PeakAmplitudesStore]: A list of peak amplitude stores.
        """
        cache_dir = self.cache_dir / "site_amplitudes"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stores = []
        for file in cache_dir.glob("*.json"):
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
        """
        Get the peak amplitude store for the given selector.

        Args:
            selector (PeakAmplitudesSelector): The selector to use.

        Returns:
            PeakAmplitudesStore: The peak amplitude store.
        """
        for store in self.get_cached_stores(selector.gf_store_id, selector.quantity):
            if store.is_suited(selector):
                return store
        return PeakAmplitudesStore.from_selector(selector)

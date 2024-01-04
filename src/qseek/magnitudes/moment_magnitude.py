from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import numpy as np
import pyrocko.moment_tensor as pmt
from pydantic import BaseModel, Field, PositiveFloat, PrivateAttr
from pyrocko import gf
from typing_extensions import Self

from qseek.magnitudes.base import EventMagnitudeCalculator
from qseek.utils import MeasurementUnit, Range

if TYPE_CHECKING:
    from pyrocko.trace import Trace

PeakAmplitude = Literal["horizontal", "vertical", "absolute"]
KM = 1e3


def norm_traces(traces: list[Trace], components: str = "ZRT") -> Trace:
    """
    Normalize traces channels.

    Args:
        traces (list[Trace]): A list of traces to normalize.
        components (str): The components to normalize.

    Returns:
        Trace: The normalized trace.
    """
    trace_selection = [tr for tr in traces if tr.channel in components]
    if not trace_selection:
        raise ValueError("No traces to normalize.")
    if len(trace_selection) == 1:
        tr = trace_selection[0].copy()
        tr.ydata = np.abs(tr.ydata)
        return tr
    data = np.array([tr.ydata for tr in trace_selection])
    norm = np.linalg.norm(data, axis=0)
    trace = traces[0].copy()
    trace.ydata = norm
    return trace


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
            peak_horizontal=norm_traces(traces, components="RT").ydata.max(),
            peak_vertical=norm_traces(traces, components="Z").ydata.max(),
            peak_absolute=norm_traces(traces).ydata.max(),
        )


class SiteAmplitudeCollection(BaseModel):
    source_depth: float
    site_amplitudes: list[SiteAmplitude] = Field(default_factory=list)

    _distances: np.ndarray = PrivateAttr()
    _horizontal: np.ndarray = PrivateAttr()
    _vertical: np.ndarray = PrivateAttr()
    _absolute: np.ndarray = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._distances = np.array([sa.distance for sa in self.site_amplitudes])
        self._horizontal = np.array([sa.peak_horizontal for sa in self.site_amplitudes])
        self._vertical = np.array([sa.peak_vertical for sa in self.site_amplitudes])
        self._absolute = np.array([sa.peak_absolute for sa in self.site_amplitudes])

    def get_amplitudes_in_range(
        self,
        distance: float,
        range: float,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> np.ndarray:
        idx = np.where(
            (self._distances >= distance - range)
            & (self._distances <= distance + range)
        )
        match peak_amplitude:
            case "horizontal":
                return self._horizontal[idx]
            case "vertical":
                return self._vertical[idx]
            case "absolute":
                return self._absolute[idx]
        raise ValueError(f"Unknown peak amplitude type {peak_amplitude}.")

    def get_amplitude(
        self,
        distance: float,
        range: float,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> float:
        amplitudes = self.get_amplitudes_in_range(distance, range, peak_amplitude)
        return float(np.median(amplitudes))

    def get_std(
        self,
        distance: float,
        range: float,
        peak_amplitude: PeakAmplitude = "absolute",
    ) -> float:
        amplitudes = self.get_amplitudes_in_range(distance, range, peak_amplitude)
        return float(np.std(amplitudes))

    def fill(self, receivers: list[gf.Receiver], traces: list[list[Trace]]) -> None:
        for receiver, rcv_traces in zip(receivers, traces, strict=True):
            self.site_amplitudes.append(SiteAmplitude.from_traces(receiver, rcv_traces))


class PeakAmplitudesStore(BaseModel):
    store_id: str = Field(
        default="moment_magnitude",
        description="Pyrocko Store ID for peak amplitude models.",
    )
    quantity: MeasurementUnit = Field(
        default="displacement",
        description="Quantity for the peak amplitude.",
    )
    frequency_min: PositiveFloat = Field(
        default=0.1,
        description="Minimum frequency for the peak amplitude.",
    )
    frequency_max: PositiveFloat = Field(
        default=10.0,
        description="Maximum frequency for the peak amplitude.",
    )
    reference_magnitude: float = Field(
        default=1.0,
        ge=-1.0,
        le=9.0,
        description="Reference magnitude in Mw.",
    )
    distance_range: Range = Field(
        default=(0 * KM, 100.0 * KM),
        description="Distance range in km. "
        "If None the whole extent of the octree is used.",
    )
    rupture_velocities: Range = Field(
        default=(0.9, 1.0),
        description="Rupture velocity range as fraction of the shear wave velocity.",
    )
    stress_drops: Range = Field(
        default=(1.0e6, 10.0e6),
        description="Stress drop range in MPa.",
    )

    site_amplitudes: list[SiteAmplitudeCollection] = Field(
        default_factory=list,
        description="Site amplitudes per source depth.",
    )

    timing_min: ClassVar[gf.Timing] = gf.Timing.T(default="vel:8")
    timing_max: ClassVar[gf.Timing] = gf.Timing.T(default="vel:2")

    _rng: np.random.Generator = PrivateAttr(default_factory=np.random.default_rng)

    def get_random_source(self, depth: float) -> gf.MTSource:
        """
        Generates a random seismic source with the given depth.

        Args:
            depth (float): The depth of the seismic source.

        Returns:
            gf.MTSource: A random moment tensor source.
        """
        rng = self._rng
        stress_drop = rng.uniform(*self.stress_drops)
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

    def get_receivers(self, n_receivers: int) -> list[gf.Receiver]:
        """
        Generate a list of receivers with random angles and distances.

        Args:
            n_receivers (int): The number of receivers to generate.

        Returns:
            list[gf.Receiver]: A list of receivers with random angles and distances.
        """
        rng = self._rng
        angles = rng.uniform(0.0, 360.0, size=n_receivers)
        distances = np.exp(rng.uniform(*np.log(self.distance_range), size=n_receivers))
        receivers: list[gf.Receiver] = []

        for i_receiver, (angle, distance) in enumerate(
            zip(angles, distances, strict=True)
        ):
            for component in "ZRT":
                receiver = gf.Receiver(
                    quantity=self.quantity,
                    store_id=self.store_id,
                    depth=0.0,
                    north_shift=distance * np.cos(np.radians(angle)),
                    east_shift=distance * np.sin(np.radians(angle)),
                    codes=("", f"{i_receiver:04d}", component),
                )
                receivers.append(receiver)
        return receivers

    async def calculate_amplitudes(
        self,
        source_depth: float,
        n_receivers: int = 1000,
    ) -> None:
        engine = gf.LocalEngine(
            use_config=True,
            store_superdirs=["."],
        )
        source = self.get_random_source(source_depth)
        targets = self.get_receivers(n_receivers)
        response: gf.Response = await asyncio.to_thread(engine.process(source, targets))

        receivers = []
        receiver_traces = []
        for _, target, traces in response.iter_results():
            for tr in traces:
                if self.frequency_min is not None:
                    tr.highpass(4, self.frequency_min, demean=False)
                if self.frequency_max is not None:
                    tr.lowpass(4, self.frequency_max, demean=False)

                if self.timing_min and self.timing_max:
                    store = engine.get_store(self.store_id)
                    tmin = store.t(self.timing_min, source, target)
                    tmax = store.t(self.timing_max, source, target)
                    if tmin is None or tmax is None:
                        raise EnvironmentError(
                            "timing determination failed (phase unavailable?)"
                        )
                    tr.chop(tmin, tmax)
            receivers.append(target)
            receiver_traces.append(traces)

        site_amplitudes = SiteAmplitudeCollection(**self.model_dump())
        site_amplitudes.fill(receivers, receiver_traces)
        self.site_amplitudes.append(site_amplitudes)


class PeakAmpliutdesSelector(PeakAmplitudesStore):
    nslc_id: str = Field(
        default="*",
        description="NSLC selector for the.",
    )


class MomentMagnitudeExtractor(EventMagnitudeCalculator):
    magnitude: Literal["moment_magnitude"] = "moment_magnitude"

    estimators: list[PeakAmpliutdesSelector] = [PeakAmpliutdesSelector()]

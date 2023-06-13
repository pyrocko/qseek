from __future__ import annotations

import logging
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, cast
from uuid import UUID, uuid4

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Extra, Field

from lassie.models.detection import EventDetection, PhaseDetection, Receiver
from lassie.models.location import Location
from lassie.models.station import Station
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from typing_extensions import Self

    from lassie.models.detection import Detections

logger = logging.getLogger(__name__)

ArrivalWeighting = Literal[
    "none", "PhaseNet", "EventSemblance", "PhaseNet+EventSemblance"
]


class EventCorrection(Location):
    uid: UUID = Field(default_factory=uuid4)
    time: datetime
    semblance: float
    distance_border: float

    phase_arrivals: dict[PhaseDescription, PhaseDetection] = {}

    def phases(self) -> tuple[PhaseDescription, ...]:
        return tuple(self.phase_arrivals.keys())


class StationCorrection(BaseModel):
    station: Station
    events: list[EventCorrection] = []

    class Config:
        extra = Extra.ignore

    def add_event(self, event: EventCorrection) -> None:
        self.events.append(event)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def has_corrections(self) -> bool:
        return bool(self.n_events)

    def iter_events(
        self, phase: PhaseDescription, observed_only: bool = True
    ) -> Iterable[EventCorrection]:
        for event in self.events:
            phase_detection = event.phase_arrivals.get(phase)
            if not phase_detection:
                continue
            if observed_only and not phase_detection.observed:
                continue
            yield event

    def phases(self) -> set[PhaseDescription]:
        return set(chain.from_iterable(event.phases() for event in self.events))

    def get_traveltime_delays(self, phase: PhaseDescription) -> np.ndarray:
        return np.fromiter(
            (
                phase.traveltime_delay.total_seconds()
                for phase in self.iter_phase_arrivals(phase=phase)
                if phase.traveltime_delay is not None
            ),
            float,
        )

    def get_event_semblances(
        self, phase: PhaseDescription, normalize: bool = False
    ) -> np.ndarray:
        values = np.fromiter(
            (event.semblance for event in self.iter_events(phase=phase)),
            float,
        )
        if normalize:
            values /= values.max()
        return values

    def get_detection_values(
        self, phase: PhaseDescription, normalize: bool = False
    ) -> np.ndarray:
        values = np.fromiter(
            (
                phase.observed.detection_value
                for phase in self.iter_phase_arrivals(phase=phase)
            ),
            float,
        )
        if normalize:
            values /= values.max()
        return values

    def get_average_delay(
        self,
        phase: PhaseDescription,
        weighted: ArrivalWeighting = "none",
    ) -> float:
        """Average delay times. Weighted and unweighted.

        Args:
            phase (PhaseDescription): Name of the phase
            weighted (ArrivalWeighting, optional): Weights, choose from "none",
                "PhaseNet" and "PhaseNet+EventSemblance". Defaults to "none".

        Returns:
            float: Delay time.
        """
        weights = None
        traveltime_delays = self.get_traveltime_delays(phase)
        if not traveltime_delays.size:
            return 0.0

        if weighted == "PhaseNet":
            weights = self.get_detection_values(phase)
        elif weighted == "EventSemblance":
            weights = self.get_event_semblances(phase)
        elif weighted == "PhaseNet+EventSemblance":
            weights = self.get_detection_values(phase, normalize=True)
            weights += self.get_event_semblances(phase, normalize=True)
        return float(np.average(traveltime_delays, weights=weights))

    def get_delays_std(self, phase: PhaseDescription) -> float:
        return float(np.std(self.get_traveltime_delays(phase)))

    def iter_phase_arrivals(
        self, phase: PhaseDescription, observed_only: bool = True
    ) -> Iterable[PhaseDetection]:
        for event in self.iter_events(phase, observed_only=observed_only):
            yield event.phase_arrivals[phase]

    def get_csv_data(self) -> dict[str, float]:
        station = self.station
        data = {"lat": station.effective_lat, "lon": station.effective_lon}
        avg_delay = self.get_average_delay
        for phase in self.phases():
            data[f"{phase}:delay"] = avg_delay(phase)
            data[f"{phase}:delay_phasenet"] = avg_delay(phase, "PhaseNet")
            data[f"{phase}:delay_semblance"] = avg_delay(phase, "EventSemblance")
            data[f"{phase}:delay_phasenet_semblance"] = avg_delay(
                phase, "PhaseNet+EventSemblance"
            )
        return data

    @classmethod
    def from_receiver(cls, receiver: Receiver) -> Self:
        return cls(station=Station.parse_obj(receiver))

    def plot(self, filename: Path) -> None:
        phases = self.phases()
        n_phases = len(phases)
        fig, axes = plt.subplots(4, n_phases)
        axes = axes.T
        fig.set_size_inches(8, 12)

        for ax_col, phase in zip(axes, phases, strict=True):
            delays = self.get_traveltime_delays(phase=phase)
            n_delays = len(delays)
            events = [event for event in self.iter_events(phase)]
            if not events:
                continue

            ax_raw = ax_col[0]
            ax_raw.set_title(f"{phase} ({n_delays} picks)")
            ax_raw.hist(delays)
            ax_raw.axvline(
                self.get_average_delay(phase),
                ls="--",
                c="k",
            )

            ax_phase_net = ax_col[1]
            ax_phase_net.set_title("PhaseNet", fontsize="small")
            ax_phase_net.hist(delays, weights=self.get_detection_values(phase), bins=8)
            ax_phase_net.axvline(
                self.get_average_delay(phase, weighted="PhaseNet"), ls="--", c="k"
            )

            ax_event_semblance = ax_col[2]
            ax_event_semblance.set_title("Event Semblance", fontsize="small")
            ax_event_semblance.hist(
                delays, weights=self.get_event_semblances(phase), bins=8
            )
            ax_event_semblance.axvline(
                self.get_average_delay(phase, weighted="EventSemblance"), ls="--", c="k"
            )

            ax_pn_es = ax_col[3]
            ax_pn_es.set_title("PhaseNet + Event Semblance", fontsize="small")
            ax_pn_es.hist(
                delays,
                weights=self.get_detection_values(phase, normalize=True)
                + self.get_event_semblances(phase, normalize=True),
                bins=8,
            )
            ax_pn_es.axvline(
                self.get_average_delay(phase, weighted="PhaseNet+EventSemblance"),
                ls="--",
                c="k",
            )

            for ax in ax_col:
                ax = cast(plt.Axes, ax)
                ax.axvline(0.0, c="k")
                xlim = max(np.abs(ax.get_xlim()))
                ax.set_xlim(-xlim, xlim)
                ax.grid(alpha=0.3)

            ax_col[-1].set_xlabel("Time Residual [s]")

        logger.info("saving residual plot to %s", filename)
        fig.savefig(str(filename))
        plt.close()


class StationCorrections(BaseModel):
    station_corrections: dict[tuple[str, str, str], StationCorrection] = {}

    def scan_detections(self, detections: Detections) -> None:
        for event in detections:
            self.add_event(event)

    def add_event(self, event: EventDetection) -> None:
        if event.in_bounds:
            return

        logger.info("loading event %s", event.time)
        for receiver in event.receivers:
            try:
                sta_correction = self.get_station_correction(receiver.nsl)
            except KeyError:
                sta_correction = StationCorrection.from_receiver(receiver)
                self.station_corrections[receiver.nsl] = sta_correction

            sta_correction.add_event(
                EventCorrection.construct(
                    phase_arrivals=receiver.phase_arrivals,
                    **event.dict(),
                )
            )

    def get_station_correction(self, nsl: tuple[str, str, str]) -> StationCorrection:
        return self.station_corrections[nsl]

    def get_correction(
        self, phase: PhaseDescription, station_nsl: tuple[str, str, str]
    ) -> float:
        ...

    def save_plots(self, folder: Path) -> None:
        folder.mkdir(exist_ok=True)
        for correction in self.station_corrections.values():
            correction.plot(
                filename=folder / f"corrections-{correction.station.pretty_nsl}.png"
            )

    def save_csv(self, filename: Path) -> None:
        logger.info("writing corrections to %s", filename)
        csv_data = [correction.get_csv_data() for correction in self]
        columns = set(chain.from_iterable(data.keys() for data in csv_data))
        with filename.open("w") as file:
            file.write(f"{', '.join(columns)}\n")
            for data in csv_data:
                file.write(
                    f"{', '.join(str(data.get(key, -9999.9)) for key in columns)}\n"
                )

    def __iter__(self) -> Iterator[StationCorrection]:
        return iter(self.station_corrections.values())

    @classmethod
    def from_detections(cls, detections: Detections) -> Self:
        logger.info("loading detections")
        station_corrections = cls()
        station_corrections.scan_detections(detections=detections)
        return station_corrections

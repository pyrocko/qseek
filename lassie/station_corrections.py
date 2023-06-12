from __future__ import annotations

import logging
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Self, cast
from uuid import UUID, uuid4

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Extra, Field

from lassie.models.detection import EventDetection, PhaseDetection, Receiver
from lassie.models.location import Location
from lassie.models.station import Station
from lassie.utils import PhaseDescription

if TYPE_CHECKING:
    from lassie.models.detection import Detections

logger = logging.getLogger(__name__)


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

    def phases(self) -> set[PhaseDescription]:
        return set(chain.from_iterable(event.phases() for event in self.events))

    @property
    def n_events(self) -> int:
        return len(self.events)

    def aggregate_traveltimes(
        self,
        phase: PhaseDescription,
        aggregator: Callable[[np.ndarray], float] = np.average,
    ) -> float:
        return aggregator(self.get_traveltime_delays(phase=phase))

    def add_event(self, event: EventCorrection) -> None:
        self.events.append(event)

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

    def get_traveltime_delays(self, phase: PhaseDescription) -> np.ndarray:
        return np.fromiter(
            (
                phase.traveltime_delay.total_seconds()
                for phase in self.iter_phase_detection(phase=phase)
                if phase.traveltime_delay is not None
            ),
            float,
        )

    def iter_phase_detection(
        self, phase: PhaseDescription, observed_only: bool = True
    ) -> Iterable[PhaseDetection]:
        for event in self.iter_events(phase, observed_only=observed_only):
            yield event.phase_arrivals[phase]

    @classmethod
    def from_receiver(cls, receiver: Receiver) -> Self:
        return cls(station=Station.parse_obj(receiver))

    def plot(self, filename: Path) -> None:
        phases = self.phases()
        n_phases = len(phases)
        fig, axes = plt.subplots(4, n_phases)
        axes = axes.T
        fig.set_size_inches(8, 12)

        for ax_col, phase in zip(axes, phases):
            delays = self.get_traveltime_delays(phase=phase)
            n_delays = len(delays)
            events = [event for event in self.iter_events(phase)]
            if not events:
                continue

            event_semblance = np.array([event.semblance for event in events])
            try:
                event_semblance /= event_semblance.max()
            except ValueError:
                print(event_semblance, [event for event in self.iter_events(phase)])
            phase_net_detection_values = np.array(
                [
                    event.phase_arrivals[phase].observed.detection_value
                    for event in events
                ]
            )
            phase_net_detection_values /= phase_net_detection_values.max()

            ax_raw = ax_col[0]
            ax_raw.set_title(f"{phase} ({n_delays} picks)")
            ax_raw.hist(delays)
            ax_raw.axvline(
                np.average(delays),
                ls="--",
                c="k",
            )

            ax_phase_net = ax_col[1]
            ax_phase_net.set_title("PhaseNet", fontsize="small")
            ax_phase_net.hist(
                delays,
                weights=phase_net_detection_values,
                bins=8,
            )
            try:
                ax_phase_net.axvline(
                    np.average(delays, weights=phase_net_detection_values),
                    ls="--",
                    c="k",
                )
            except ZeroDivisionError:
                print(phase_net_detection_values)
                print(self)

            ax_event_semblance = ax_col[2]
            ax_event_semblance.set_title("Event Semblance", fontsize="small")
            ax_event_semblance.hist(
                delays,
                weights=event_semblance,
                bins=8,
            )
            ax_event_semblance.axvline(
                np.average(delays, weights=event_semblance),
                ls="--",
                c="k",
            )

            ax_pn_es = ax_col[3]
            ax_pn_es.set_title("PhaseNet + Event Semblance", fontsize="small")
            ax_pn_es.hist(
                delays,
                weights=phase_net_detection_values + event_semblance,
                bins=8,
            )
            try:
                ax_pn_es.axvline(
                    np.average(
                        delays, weights=phase_net_detection_values + event_semblance
                    ),
                    ls="--",
                    c="k",
                )
            except ZeroDivisionError:
                print(self)

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
        if not event.in_bounds:
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

    def plot_map(self, filename: Path) -> None:
        fig = plt.figure()
        ax = fig.gca()

        lat_lon = [
            corr.station.effective_lat_lon for corr in self.station_corrections.values()
        ]
        ax.scatter(*lat_lon)
        fig.savefig(str(filename))
        plt.close()

    @classmethod
    def from_detections(cls, detections: Detections) -> Self:
        logger.info("loading detections")
        station_corrections = cls()
        station_corrections.scan_detections(detections=detections)
        return station_corrections

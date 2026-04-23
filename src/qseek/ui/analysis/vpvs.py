from typing import Literal, NamedTuple, Self

import numpy as np

from qseek.models.detection import EventDetection, PhaseDetection
from qseek.models.location import Location


class PSTravelTime(NamedTuple):
    travel_time_p: float
    travel_time_s: float

    confidence_p: float
    confidence_s: float

    event: EventDetection
    receiver: Location

    @classmethod
    def from_arrivals(
        cls,
        arrival_p: PhaseDetection,
        arrival_s: PhaseDetection,
        event: EventDetection,
        receiver: Location,
    ) -> Self:
        origin_time = event.time.timestamp()
        travel_time_p = arrival_p.observed.time.timestamp() - origin_time
        travel_time_s = arrival_s.observed.time.timestamp() - origin_time
        return cls(
            travel_time_p=travel_time_p,
            travel_time_s=travel_time_s,
            confidence_p=arrival_p.observed.detection_value,
            confidence_s=arrival_s.observed.detection_value,
            event=event,
            receiver=receiver,
        )


class PSCollection:
    travel_times: list[PSTravelTime]

    def __init__(self):
        self.travel_times = []

    def get_travel_times(self, phase: Literal["P", "S"]) -> np.ndarray:
        if phase == "P":
            return np.array([tt.travel_time_p for tt in self.travel_times])
        return np.array([tt.travel_time_s for tt in self.travel_times])

    def get_confidences(self, phase: Literal["P", "S"]) -> np.ndarray:
        if phase == "P":
            return np.array([tt.confidence_p for tt in self.travel_times])
        return np.array([tt.confidence_s for tt in self.travel_times])

    def add_event(self, event: EventDetection):
        available_phases = event.receivers.get_available_phases()
        if {ph[-1] for ph in available_phases} != {"P", "S"}:
            return

        append = self.travel_times.append

        for rcv in event.receivers:
            p_key = next((ph for ph in rcv.phase_arrivals if ph.endswith("P")), None)
            s_key = next((ph for ph in rcv.phase_arrivals if ph.endswith("S")), None)
            if p_key is None or s_key is None:
                continue

            p_arrival = rcv.phase_arrivals[p_key]
            s_arrival = rcv.phase_arrivals[s_key]
            if p_arrival.observed is None or s_arrival.observed is None:
                continue

            append(
                PSTravelTime.from_arrivals(
                    arrival_p=p_arrival, arrival_s=s_arrival, event=event, receiver=rcv
                )
            )

from __future__ import annotations

import logging
import math
from itertools import combinations
from pathlib import Path

from rich import progress

from qseek.exporters.base import Exporter
from qseek.exporters.cross_correlation.model import EventCorrelationPair
from qseek.models.detection import EventDetection, Receiver
from qseek.search import Search

logger = logging.getLogger(__name__)

KM = 1e3


def _confident_receivers(
    event: EventDetection,
    phase: str,
    min_confidence: float,
) -> list[Receiver]:
    """Get receivers of an event with a confident observed arrival for a phase."""
    receivers = []
    for receiver in event.receivers:
        arrival = receiver.phase_arrivals.get(phase)
        if arrival is None or arrival.observed is None:
            continue
        if arrival.observed.detection_value < min_confidence:
            continue
        receivers.append(receiver)
    return receivers


class TravelTimeCrossCorrelation(Exporter):
    cross_correlation_threshold: float = 0.5
    min_pick_confidence: float = 0.3
    min_event_semblance: float = 0.5
    max_event_distance: float = 2.0 * KM
    seconds_before: float = 1.0
    seconds_after: float = 2.0

    async def export(
        self,
        rundir: Path,
        outdir: Path,
    ) -> Path:
        logger.info("Cross-correlating travel time differences for nearby event pairs.")

        search = Search.load_rundir(rundir)
        catalog = search.catalog
        waveform_provider = search.data_provider
        phases = search.image_functions.get_phases()

        events = [ev for ev in catalog if ev.semblance >= self.min_event_semblance]
        total_pairs = math.comb(len(events), 2) if len(events) > 1 else 0

        cross_correlations: list[EventCorrelationPair] = []
        n_candidate_pairs = 0

        for event_a, event_b in progress.track(
            combinations(events, 2),
            description="Cross-correlating event pairs",
            total=total_pairs,
        ):
            if event_a.distance_to(event_b) > self.max_event_distance:
                continue
            n_candidate_pairs += 1

            for phase in phases:
                receivers_a = _confident_receivers(
                    event_a,
                    phase,
                    self.min_pick_confidence,
                )
                receivers_b = _confident_receivers(
                    event_b,
                    phase,
                    self.min_pick_confidence,
                )
                if not receivers_a or not receivers_b:
                    continue

                traces_a = await event_a.receivers.get_waveforms(
                    waveform_provider,
                    seconds_before=self.seconds_before,
                    seconds_after=self.seconds_after,
                    phase=phase,
                    receivers=receivers_a,
                    picked_only=True,
                    want_incomplete=False,
                )
                traces_b = await event_b.receivers.get_waveforms(
                    waveform_provider,
                    seconds_before=self.seconds_before,
                    seconds_after=self.seconds_after,
                    phase=phase,
                    receivers=receivers_b,
                    picked_only=True,
                    want_incomplete=False,
                )
                if not traces_a or not traces_b:
                    continue

                try:
                    cc = await EventCorrelationPair.calculate(
                        event_a, event_b, traces_a, traces_b, phase
                    )
                except ValueError:
                    continue

                cross_correlations.append(cc)

        logger.info(
            "Cross-correlated %d station-phase observations across %d event pairs"
            " (%d candidates within %.0f m of %d qualifying events).",
            len(cross_correlations),
            n_candidate_pairs,
            total_pairs,
            self.max_event_distance,
            len(events),
        )

        return outdir

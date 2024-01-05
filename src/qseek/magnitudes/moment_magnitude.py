from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import DirectoryPath, Field, PositiveFloat, PrivateAttr
from typing_extensions import Self

from qseek.magnitudes.base import EventMagnitude, EventMagnitudeCalculator
from qseek.magnitudes.moment_magnitude_store import (
    PeakAmplitudesBase,
    PeakAmplitudesStore,
)
from qseek.utils import CACHE_DIR, MeasurementUnit

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel
    from pyrocko.trace import Trace

    from qseek.models.detection import EventDetection, Receiver
    from qseek.models.station import Stations
    from qseek.octree import Octree


logger = logging.getLogger(__name__)

KM = 1e3


class MomentMagnitude(EventMagnitude):
    magnitude: Literal["MomentMagnitude"] = "MomentMagnitude"

    average: float = Field(
        default=0.0,
        description="Average moment magnitude.",
    )
    error: float = Field(
        default=0.0,
        description="Average error of moment magnitude.",
    )

    @classmethod
    def from_store(
        cls,
        parent: MomentMagnitudeExtractor,
        event: EventDetection,
        receivers: list[Receiver],
        traces: list[Trace],
    ) -> Self:
        raise NotImplementedError


"""
Problems:
* Set the measurement unit for the traces in the selector.
* Set the distance range for the selector.
"""


class MomentMagnitudeExtractor(EventMagnitudeCalculator):
    magnitude: Literal["MomentMagnitude"] = "MomentMagnitude"

    seconds_before: PositiveFloat = 10.0
    seconds_after: PositiveFloat = 10.0
    padding_seconds: PositiveFloat = 10.0
    quantity: MeasurementUnit = "displacement"

    gf_store_dirs: list[DirectoryPath] = [Path(".")]
    models: list[PeakAmplitudesBase] = [PeakAmplitudesBase()]

    _stores: list[PeakAmplitudesStore] = PrivateAttr(default_factory=list)

    def load_cached_amplitude_stores(self) -> list[PeakAmplitudesStore]:
        """
        Load the peak amplitude stores from the cache directory.

        Returns:
            A list of PeakAmplitudesStores
        """
        cache_dir = CACHE_DIR / "site_amplitudes"
        cache_dir.mkdir(parents=True, exist_ok=True)
        gf_store_ids = [s.gf_store_id for s in self.models]
        stores = []
        for file in cache_dir.glob("*.json"):
            store_id, _ = file.stem.split("-")
            if store_id not in gf_store_ids:
                continue
            store = PeakAmplitudesStore.model_validate_json(file.read_text())
            stores.append(store)
        logger.info("loaded %d cached amplitude stores", len(stores))
        return stores

    async def prepare(self, octree: Octree, stations: Stations) -> None:
        """
        Prepare the moment magnitude calculation

        The distance and depth ranges are set for each model, loading cached
        stores if available, and calculating site amplitudes if necessary.

        Args:
            octree (Octree): The octree representing the seismic event.
            stations (Stations): The stations associated with the seismic event.
        """
        ...

    async def add_magnitude(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        traces = await event.receivers.get_waveforms_restituted(
            squirrel,
            quantity=self.quantity,
            seconds_before=self.seconds_before,
            seconds_after=self.seconds_after,
            demean=True,
            cut_off_fade=True,
            seconds_fade=self.padding_seconds,
        )
        if not traces:
            logger.warning("No restituted traces found for event %s", event.time)
            return

        grouped_traces = []
        receivers = []
        for nsl, grp_traces in itertools.groupby(traces, key=lambda tr: tr.nslc_id[:3]):
            grouped_traces.append(list(grp_traces))
            receivers.append(event.receivers.get_receiver(nsl))

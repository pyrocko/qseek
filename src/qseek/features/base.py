from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection
    from qseek.models.station import StationInventory
    from qseek.waveforms.base import WaveformProvider


class ReceiverFeature(BaseModel):
    feature: Literal["ReceiverFeature"] = "ReceiverFeature"

    @classmethod
    def get_subclasses(cls) -> tuple[type[ReceiverFeature], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())


class EventFeature(BaseModel):
    feature: Literal["EventFeature"] = "EventFeature"

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventFeature], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())


class FeatureExtractor(BaseModel):
    feature: Literal["FeatureExtractor"] = "FeatureExtractor"

    @classmethod
    def get_subclasses(cls) -> tuple[type[FeatureExtractor], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())

    async def add_features(
        self,
        waveform_provider: WaveformProvider,
        stations: StationInventory,
        event: EventDetection,
    ) -> None:
        raise NotImplementedError

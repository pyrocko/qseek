from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection


class ReceiverFeature(BaseModel):
    feature: Literal["Feature"] = "Feature"

    @classmethod
    def get_subclasses(cls) -> tuple[type[ReceiverFeature], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        subclasses = list(cls.__subclasses__())
        if len(subclasses) == 1:
            subclasses.append(cls)
        return tuple(subclasses)


class EventFeature(BaseModel):
    feature: Literal["Feature"] = "Feature"

    @classmethod
    def get_subclasses(cls) -> tuple[type[EventFeature], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        subclasses = list(cls.__subclasses__())
        if len(subclasses) == 1:
            subclasses.append(cls)
        return tuple(subclasses)


class FeatureExtractor(BaseModel):
    feature: Literal["FeatureExtractor"] = "FeatureExtractor"

    @classmethod
    def get_subclasses(cls) -> tuple[type[FeatureExtractor], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        subclasses = list(cls.__subclasses__())
        if len(subclasses) == 1:
            subclasses.append(cls)
        return tuple(subclasses)

    async def add_features(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        raise NotImplementedError

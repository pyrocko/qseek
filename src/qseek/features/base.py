from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from qseek.models.detection import EventDetection


class ReceiverFeature(BaseModel):
    feature: Literal["Feature"] = "Feature"


class EventFeature(BaseModel):
    feature: Literal["Feature"] = "Feature"


class FeatureExtractor(BaseModel):
    feature: Literal["FeatureExtractor"] = "FeatureExtractor"

    async def add_features(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> None:
        raise NotImplementedError

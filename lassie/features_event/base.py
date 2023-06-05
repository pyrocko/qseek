from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from lassie.models.detection import EventDetection


class EventFeature(BaseModel):
    feature: Literal["Feature"] = "Feature"


class EventFeatureExtractor(BaseModel):
    name: Literal["FeatureExtractor"] = "FeatureExtractor"

    async def get_features(
        self,
        squirrel: Squirrel,
        event: EventDetection,
    ) -> EventFeature:
        raise NotImplementedError

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from lassie.models.detection import EventDetection


class Feature(BaseModel):
    name: Literal["Feature"] = "Feature"


class ReceiverFeatureExtractor(BaseModel):
    name: Literal["FeatureExtractor"] = "FeatureExtractor"

    def get_features(self, squirrel: Squirrel, detection: EventDetection) -> Feature:
        raise NotImplementedError

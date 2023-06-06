from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyrocko.squirrel import Squirrel

    from lassie.models.detection import PhaseDetection


class ReceiverFeature(BaseModel):
    feature: Literal["Feature"] = "Feature"


class ReceiverFeatureExtractor(BaseModel):
    feature: Literal["FeatureExtractor"] = "FeatureExtractor"

    async def get_features(
        self,
        squirrel: Squirrel,
        phase_detection: PhaseDetection,
    ) -> list[ReceiverFeature | None]:
        raise NotImplementedError

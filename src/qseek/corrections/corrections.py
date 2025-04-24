from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Union

from pydantic import BaseModel, Field

from qseek.corrections.base import TravelTimeCorrections

# Has to be imported to register as subclass
from qseek.corrections.simple import SimpleCorrections  # noqa: F401

logger = logging.getLogger(__name__)

StationCorrectionType = Annotated[
    Union[(TravelTimeCorrections, *TravelTimeCorrections.get_subclasses())],
    Field(..., discriminator="corrections"),
]


def corrections_from_path(path: Path) -> TravelTimeCorrections:
    class TempModel(BaseModel):
        corrections: StationCorrectionType

    correction_file = path / "corrections.json"
    json = f'{{"corrections": {correction_file.read_text()}}}'
    data = TempModel.model_validate(json)
    logger.info("loaded corrections from %s", path)
    return data.corrections

from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, PositiveFloat, confloat, validator
from pyrocko.squirrel import Squirrel

from lassie.images import ImageFunctions, PhaseNet
from lassie.models import Stations
from lassie.octree import Octree
from lassie.tracers import RayTracers
from lassie.tracers.constant_velocity import ConstantVelocityTracer


class Config(BaseModel):
    squirrel_environment: Path = Path(".")
    waveform_data: list[Path]

    time_span: tuple[datetime | None, datetime | None] = (None, None)
    sampling_rate: confloat(ge=5.0, le=30) = 20.0
    detection_threshold: PositiveFloat = 0.1

    stations: Stations
    octree: Octree = Octree()

    image_functions: ImageFunctions = ImageFunctions(__root__=[PhaseNet()])
    ray_tracers: RayTracers = RayTracers(__root__=[ConstantVelocityTracer()])

    @validator("time_span")
    def _validate_time_span(cls, range):  # noqa: N805
        if range[0] >= range[1]:
            raise ValueError(f"time range is invalid {range[0]} - {range[1]}")
        return range

    @property
    def start_time(self) -> datetime | None:
        return self.time_span[0]

    @property
    def end_time(self) -> datetime | None:
        return self.time_span[1]

    def get_squirrel(self) -> Squirrel:
        squirrel = Squirrel(str(self.squirrel_environment))
        paths = []
        for path in self.waveform_data:
            if "**" in str(path):
                paths.extend(glob.glob(str(path), recursive=True))
            else:
                paths.append(str(path))
        squirrel.add(paths, check=False)
        return squirrel

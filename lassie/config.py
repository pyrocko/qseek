from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, validator
from pyrocko.squirrel import Squirrel

from lassie.images import ImageFunctions, PhaseNet
from lassie.models import Stations
from lassie.octree import Octree
from lassie.tracers import ConstantVelocityTracer, RayTracers


class Config(BaseModel):
    stations: Stations = Stations()

    squirrel_environment: Path = Path(".")
    waveform_data: list[Path]

    time_span: tuple[datetime, datetime] = (
        datetime.fromisoformat("2023-04-11T00:00:00+00:00"),
        datetime.fromisoformat("2023-04-18T00:00:00+00:00"),
    )

    ray_tracers: RayTracers = RayTracers(tracers=[ConstantVelocityTracer()])
    image_functions: ImageFunctions = ImageFunctions(functions=[PhaseNet()])

    octree: Octree = Octree()

    @validator("time_span")
    def _validate_time_span(cls, range):  # noqa: N805
        if range[0] < range[1]:
            raise ValueError(f"Time range is invalid {range[0]} - {range[1]}")
        return range

    @property
    def start_time(self) -> datetime:
        return self.time_span[0]

    @property
    def end_time(self) -> datetime:
        return self.time_span[1]

    def get_squirrel(self) -> Squirrel:
        squirrel = Squirrel(str(self.squirrel_environment))
        paths = []
        for path in self.waveform_data:
            if "**" in str(path):
                paths.extend(glob.glob(str(path)))
            else:
                paths.append(str(path))
        squirrel.add(paths, check=False)
        return squirrel

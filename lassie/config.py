import glob
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, constr, validator
from pyrocko.model import Station, load_stations
from pyrocko.squirrel import Squirrel

from lassie.images import PhaseNet
from lassie.images.base import ImageFunction
from lassie.models import Receivers
from lassie.octree import Octree
from lassie.tracers import ConstantVelocityTracer, RayTracers

NSL_RE = r"^[a-zA-Z0-9]{0,2}\.[a-zA-Z0-9]{0,5}\.[a-zA-Z0-9]{0,3}$"


class Config(BaseModel):
    stations_file: Path = Path("stations.yaml")
    station_blacklist: list[constr(regex=NSL_RE)] = ["NE.STA.LOC"]

    squirrel_environment: Path = Path(".")
    waveform_data: list[Path] = [Path("data/")]

    time_span: tuple[datetime, datetime] = (
        datetime.fromisoformat("2023-04-11T00:00:00Z"),
        datetime.fromisoformat("2023-04-18T00:00:00Z"),
    )

    image_functions: list[ImageFunction] = [PhaseNet()]
    ray_tracers: RayTracers = RayTracers(__root__=[ConstantVelocityTracer()])

    octree: Octree = Octree()

    @validator("time_span")
    def _validate_time_span(cls, range):  # noqa: N805
        assert range[0] < range[1]
        return range

    @validator("waveform_data")
    def _validate_data_paths(cls, paths: list[Path]) -> list[Path]:  # noqa: N805
        for path in paths:
            if "**" in str(path):
                continue
            if not path.exists():
                raise FileNotFoundError(f"Cannot find data path {path}")
        return paths

    @property
    def cache_path(self) -> Path:
        cache = Path("cache")
        if not cache.exists():
            cache.mkdir()
        return cache

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

    def get_receivers(self) -> Receivers:
        stations = load_stations(self.stations_file)

        def in_blacklist(station: Station) -> bool:
            if ".".join(station.nsl()) in self.station_blacklist:
                return False
            return True

        return Receivers.from_pyrocko_stations(filter(in_blacklist, stations))

    @property
    def start_time(self) -> datetime:
        return self.time_span[0]

    @property
    def end_time(self) -> datetime:
        return self.time_span[1]

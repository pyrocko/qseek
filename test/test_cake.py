from pathlib import Path
from tempfile import TemporaryDirectory

from lassie.config import Config
from lassie.models.location import Location
from lassie.tracers.cake import CakeTracer, EarthModel, SPTreeModel, Timing

km = 1e3


def test_sptree_model(sample_config: Config):
    model = SPTreeModel.new(
        earthmodel=EarthModel(),
        distance_bounds=(0 * km, 10 * km),
        receiver_depth_bounds=(0 * km, 0 * km),
        source_depth_bounds=(0 * km, 10 * km),
        spatial_tolerance=200,
        time_tolerance=0.05,
        timing=Timing(definition="P,p"),
    )

    with TemporaryDirectory() as d:
        tmp = Path(d)
        file = model.save(tmp)

        model2 = SPTreeModel.load(file)
        model2._get_sptree()

    source = Location(
        lat=0.0,
        lon=0.0,
        north_shift=1 * km,
        east_shift=1 * km,
        depth=5.0 * km,
    )
    receiver = Location(
        lat=0.0,
        lon=0.0,
        north_shift=0 * km,
        east_shift=0 * km,
        depth=0,
    )

    model.get_traveltime(source, receiver)
    model.get_traveltimes(sample_config.octree, sample_config.stations)


def test_cake_tracer(sample_config: Config):
    tracer = CakeTracer()
    tracer.prepare(sample_config.octree, sample_config.stations)

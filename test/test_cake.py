from pathlib import Path
from tempfile import TemporaryDirectory

from lassie.models.location import Location
from lassie.tracers.cake import EarthModel, Timing, TraveltimeTree

KM = 1e3


def test_sptree_model():
    model = TraveltimeTree.new(
        earthmodel=EarthModel(),
        distance_bounds=(0 * KM, 10 * KM),
        receiver_depth_bounds=(0 * KM, 0 * KM),
        source_depth_bounds=(0 * KM, 10 * KM),
        spatial_tolerance=200,
        time_tolerance=0.05,
        timing=Timing(definition="P,p"),
    )

    with TemporaryDirectory() as d:
        tmp = Path(d)
        file = model.save(tmp)

        model2 = TraveltimeTree.load(file)
        model2._load_sptree()

    source = Location(
        lat=0.0,
        lon=0.0,
        north_shift=1 * KM,
        east_shift=1 * KM,
        depth=5.0 * KM,
    )
    receiver = Location(
        lat=0.0,
        lon=0.0,
        north_shift=0 * KM,
        east_shift=0 * KM,
        depth=0,
    )

    model.get_traveltime(source, receiver)

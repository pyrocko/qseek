from lassie.config import Config
from lassie.search import Search

km = 1e3


def test_search(sample_config: Config) -> None:
    config = sample_config

    search = Search(
        octree=config.octree,
        stations=config.stations,
        image_functions=config.image_functions,
        ray_tracers=config.ray_tracers,
    )

    search.scan_squirrel(config.get_squirrel())

from pathlib import Path

from lassie.config import Config
from lassie.models.location import locations_to_csv
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
    print(config.octree.n_nodes)

    search.scan_squirrel(config.get_squirrel(), *config.time_span)
    locations = [sta for sta in search.stations]
    locations += [node.as_location() for node in search.octree]
    locations_to_csv(locations, Path("/tmp/test.csv"))

    print(search.detections)
    search.detections.to_csv(Path("/tmp/test-detections.csv"))

from lassie.config import Config
from lassie.search import Search

km = 1e3


def test_search(sample_config: Config) -> None:
    config = sample_config

    centroid = config.stations.get_centroid()
    config.octree.center_lat = centroid.lat
    config.octree.center_lon = centroid.lon
    config.octree.surface_elevation = centroid.elevation
    config.octree.east_bounds = (-20 * km, 20 * km)
    config.octree.north_bounds = (-20 * km, 20 * km)
    config.octree.init_nodes()

    search = Search(
        octree=config.octree,
        stations=config.stations,
        image_functions=config.image_functions,
        ray_tracers=config.ray_tracers,
    )
    search.scan_squirrel(config.get_squirrel())

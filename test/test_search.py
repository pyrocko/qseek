from lassie.config import Config
from lassie.search import Search

km = 1e3


def test_search(sample_config: Config) -> None:
    config = sample_config

    centroid = config.stations.get_centroid()

    # Laacher See
    config.octree.center_lat = 50.41255
    config.octree.center_lon = 7.26816
    config.octree.surface_elevation = centroid.elevation
    config.octree.east_bounds = (-30 * km, 30 * km)
    config.octree.north_bounds = (-30 * km, 30 * km)
    config.octree.depth_bounds = (0 * km, 20 * km)
    config.octree.init_nodes()

    search = Search(
        octree=config.octree,
        stations=config.stations,
        image_functions=config.image_functions,
        ray_tracers=config.ray_tracers,
    )

    search.scan_squirrel(config.get_squirrel())

from pathlib import Path

import pytest

from lassie.models.location import locations_to_csv
from lassie.search import SquirrelSearch

km = 1e3


@pytest.mark.skip(reason="Fail")
def test_search() -> None:
    search = SquirrelSearch()

    # search.scan_squirrel()
    locations = search.stations.model_copy()
    locations += [node.as_location() for node in search.octree]
    locations_to_csv(locations, Path("/tmp/test.csv"))

    print(search._detections)
    search._detections.to_csv(Path("/tmp/test-detections.csv"))

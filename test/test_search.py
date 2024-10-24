from pathlib import Path

import pytest

from qseek.models.location import locations_to_csv
from qseek.search import Search

km = 1e3


@pytest.mark.skip(reason="Fail")
def test_search() -> None:
    search = Search()

    # search.scan_squirrel()
    locations = search.stations.model_copy()
    locations += [node.as_location() for node in search.octree]
    locations_to_csv(locations, Path("/tmp/test.csv"))

    search.detections.to_csv(Path("/tmp/test-detections.csv"))

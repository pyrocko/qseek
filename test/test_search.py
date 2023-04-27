from lassie.config import Config
from lassie.search import Search


def test_search(sample_config: Config) -> None:
    config = sample_config
    search = Search(config)
    search.search()

import pytest
from pyrocko import gf
from qseek.magnitudes.moment_magnitude_store import (
    PeakAmplitudesBase,
    PeakAmplitudesStore,
)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def engine():
    return gf.LocalEngine(use_config=True)


def has_store(store_id) -> bool:
    try:
        engine().get_store(store_id)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not has_store("crust2_de"), reason="crust2_de not available")
def test_peak_amplitudes() -> None:
    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id="crust2_de",
        quantity="displacement",
    )

    PeakAmplitudesStore.from_selector(peak_amplitudes)

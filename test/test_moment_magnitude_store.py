from __future__ import annotations

import pytest
from pyrocko import gf
from qseek.magnitudes.moment_magnitude_store import (
    PeakAmplitude,
    PeakAmplitudesBase,
    PeakAmplitudesStore,
)

KM = 1e3
NM = 1e-9


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


def get_engine() -> gf.LocalEngine:
    return gf.LocalEngine(use_config=True)


@pytest.fixture(scope="module")
def engine() -> gf.LocalEngine:
    return get_engine()


def has_store(store_id) -> bool:
    try:
        get_engine().get_store(store_id)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not has_store("crust2_de"), reason="crust2_de not available")
@pytest.mark.asyncio
async def test_peak_amplitudes(engine: gf.LocalEngine) -> None:
    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id="crust2_de",
        quantity="displacement",
    )
    PeakAmplitudesStore.set_engine(engine)
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)
    await store.fill_source_depth(source_depth=2 * KM)
    await store.get_amplitude(
        source_depth=2 * KM,
        distance=10 * KM,
        n_amplitudes=10,
        max_distance=1 * KM,
        auto_fill=False,
    )


@pytest.mark.plot
@pytest.mark.asyncio
async def test_peak_amplitude_plot(engine: gf.LocalEngine) -> None:
    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id="crust2_de",
        quantity="displacement",
    )
    plot_amplitude: PeakAmplitude = "horizontal"

    PeakAmplitudesStore.set_engine(engine)
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)

    collection = await store.fill_source_depth(source_depth=2 * KM)
    collection.plot(peak_amplitude=plot_amplitude)

    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id="crust2_de",
        quantity="velocity",
    )
    PeakAmplitudesStore.set_engine(engine)
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)

    collection = await store.fill_source_depth(source_depth=2 * KM)
    collection.plot(peak_amplitude=plot_amplitude)

    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id="crust2_de",
        quantity="acceleration",
    )
    PeakAmplitudesStore.set_engine(engine)
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)

    collection = await store.fill_source_depth(source_depth=2 * KM)
    collection.plot(peak_amplitude=plot_amplitude)

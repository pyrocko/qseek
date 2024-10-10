from __future__ import annotations

import numpy as np
import pytest
from pyrocko import gf

from qseek.magnitudes.moment_magnitude_store import (
    ModelledAmplitude,
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
    await store.compute_site_amplitudes(source_depth=2 * KM, reference_magnitude=1.0)
    await store.get_amplitude_model(
        source_depth=2 * KM,
        distance=10 * KM,
        n_amplitudes=10,
        distance_cutoff=1 * KM,
        auto_fill=False,
        interpolation="nearest",
    )


@pytest.mark.skipif(
    not has_store("reykjanes_qseis"),
    reason="reykjanes_qseis not available",
)
@pytest.mark.asyncio
async def test_peak_amplitude_estimation(engine: gf.LocalEngine) -> None:
    store_id = "reykjanes_qseis"
    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id=store_id,
        quantity="displacement",
    )
    PeakAmplitudesStore.set_engine(engine)
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)
    await store.compute_site_amplitudes(source_depth=2 * KM, reference_magnitude=1.0)

    await store.find_moment_magnitude(
        source_depth=2 * KM,
        distance=10 * KM,
        observed_amplitude=0.0001,
    )


@pytest.mark.plot
@pytest.mark.asyncio
async def test_peak_amplitude_plot(engine: gf.LocalEngine) -> None:
    store_id = "reykjanes_qseis"
    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id=store_id,
        quantity="displacement",
    )
    plot_amplitude: PeakAmplitude = "absolute"

    PeakAmplitudesStore.set_engine(engine)
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)

    collection = await store.compute_site_amplitudes(
        source_depth=2 * KM, reference_magnitude=1.0
    )
    collection.plot(peak_amplitude=plot_amplitude)

    await store.find_moment_magnitude(
        source_depth=2 * KM,
        distance=10 * KM,
        observed_amplitude=0.01,
    )

    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id=store_id,
        quantity="velocity",
    )
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)

    collection = await store.compute_site_amplitudes(
        source_depth=2 * KM, reference_magnitude=2.0
    )
    collection.plot(peak_amplitude=plot_amplitude, reference_magnitude=2.0)

    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id=store_id,
        quantity="acceleration",
    )
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)

    collection = await store.compute_site_amplitudes(
        source_depth=2 * KM, reference_magnitude=1.0
    )
    collection.plot(peak_amplitude=plot_amplitude)


@pytest.mark.plot
@pytest.mark.asyncio
async def test_peak_amplitude_surface(engine: gf.LocalEngine) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LightSource
    from matplotlib.ticker import FuncFormatter

    peak_amplitudes = PeakAmplitudesBase(
        gf_store_id="crust2_de",
        quantity="velocity",
    )
    plot_amplitude: PeakAmplitude = "absolute"
    PeakAmplitudesStore.set_engine(engine)
    store = PeakAmplitudesStore.from_selector(peak_amplitudes)
    await store.fill_source_depth_range(depth_max=5 * KM)

    distances = np.linspace(0, store.max_distance, 256)
    depths = np.linspace(*store.source_depth_range, 256)

    depth_amplitudes = []
    for depth in depths:
        amplitudes: list[ModelledAmplitude] = []
        for dist in distances:
            amplitudes.append(
                await store.get_amplitude_model(
                    source_depth=depth,
                    distance=dist,
                    n_amplitudes=25,
                    reference_magnitude=1.0,
                    peak_amplitude=plot_amplitude,
                    auto_fill=False,
                )
            )
        depth_amplitudes.append(amplitudes)

    data = [[a.median for a in amplitudes] for amplitudes in depth_amplitudes]
    data = np.array(data) / NM

    fig, ax = plt.subplots()
    ls = LightSource(azdeg=315, altdeg=45)

    cmap = "viridis"
    rgb = ls.shade(np.log10(data), cmap=plt.cm.get_cmap(cmap), blend_mode="overlay")

    ax.imshow(
        rgb,
        extent=[
            0,
            store.max_distance,
            store.source_depth_range.max,
            store.source_depth_range.min,
        ],
        aspect="auto",
    )
    cm = ScalarMappable(norm=None, cmap=cmap)
    cm.set_array(data)
    cbar = fig.colorbar(cm, ax=ax)
    cbar.set_label("Absolute Displacement [nm]")

    ax.set_xlabel("Epicentral Distance [km]")
    ax.set_ylabel("Source Depth [km]")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x / KM))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: x / KM))
    plt.show()
    plt.close()

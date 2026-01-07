from __future__ import annotations

import tempfile
from pathlib import Path
from random import choices

import numpy as np
import pytest

from qseek.models.layered_model import Layer, LayeredModel
from qseek.models.location import Location
from qseek.models.station import Station, StationInventory, StationList
from qseek.octree import Octree
from qseek.tracers.cake import CakeTracer, Timing
from qseek.tracers.fast_marching import (
    FastMarchingTracer,
    FMMImplementation,
    StationTravelTimeTable,
)
from qseek.tracers.utils import (
    LayeredEarthModel1D,
    surface_distances,
    surface_distances_reference,
)
from qseek.utils import Range

KM = 1e3


def constant_earth_model() -> LayeredEarthModel1D:
    with tempfile.NamedTemporaryFile("w") as file:
        file.write(
            """ -1.0   5.0   3.0   2.7
             30.0   5.0   3.0   2.7
            """
        )
        file.flush()
        return LayeredEarthModel1D(filename=Path(file.name))


@pytest.fixture(scope="function")
def octree() -> Octree:
    return Octree(
        location=Location(
            lat=10.0,
            lon=10.0,
            elevation=0.5 * KM,
        ),
        root_node_size=2 * KM,
        n_levels=5,
        east_bounds=Range(-20 * KM, 20 * KM),
        north_bounds=Range(-20 * KM, 20 * KM),
        depth_bounds=Range(0 * KM, 10 * KM),
    )


@pytest.fixture(scope="session")
def stations() -> StationInventory:
    rng = np.random.default_rng(1232)
    n_stations = 5
    stations: list[Station] = []
    for i_sta in range(n_stations):
        station = Station(
            network="XX",
            station="STA%02d" % i_sta,
            lat=10.0,
            lon=10.0,
            elevation=rng.uniform(0, 1) * KM,
            depth=rng.uniform(0, 0.2) * KM,
            north_shift=rng.uniform(-5, 5) * KM,
            east_shift=rng.uniform(-5, 5) * KM,
        )
        stations.append(station)
    return StationInventory(stations=stations)


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", ["scikit-fmm", "pyrocko"])
async def test_station_travel_time_table_constant(
    plot: bool,
    implementation: FMMImplementation,
):
    model = LayeredModel(
        layers=[
            Layer(top_depth=0, vp=5.0 * KM, vs=3.0 * KM),
            Layer(top_depth=50 * KM, vp=5.0 * KM, vs=3.0 * KM),
        ]
    )
    station = Station(
        station="TST",
        network="TT",
        lat=0.0,
        lon=0.0,
        elevation=200.0,
    )
    table = StationTravelTimeTable(
        station=station,
        phase="fm:P",
        distance_max=20 * KM,
        depth_range=Range(0, 20 * KM),
        grid_spacing=100.0,
        earth_model=model,
    )
    await table.calculate(implementation=implementation)

    rng = np.random.default_rng()
    distances = rng.uniform(0.0, table.distance_max, 1000)
    depths = rng.uniform(table.depth_range.start, table.depth_range.end, 1000)
    await table.get_travel_times(distances, depths)

    dists = np.sqrt(distances**2 + depths**2)
    analytical_vp_tt = dists / model.layers[0].vp

    # np.testing.assert_allclose(travel_times, analytical_vp_tt)
    grid = np.meshgrid(
        table._distances,
        table._depths,
        indexing="ij",
    )
    dists = np.linalg.norm(
        np.array([grid[0].flatten(), grid[1].flatten()]),
        axis=0,
    ).reshape(*grid[0].shape)
    analytical_vp_tt = dists / model.layers[0].vp

    np.testing.assert_allclose(
        table._travel_times,
        analytical_vp_tt,
        atol=1e-1,
    )

    if plot:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        data = table._travel_times - analytical_vp_tt
        data_max = np.max(np.abs(data))
        ax.imshow(
            data,
            extent=(
                0.0,
                table.distance_max,
                table.depth_range.start,
                table.depth_range.end,
            ),
            cmap="seismic",
            vmin=-data_max,
            vmax=data_max,
            aspect="equal",
            origin="lower",
        )

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        plt.show()


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", ["scikit-fmm", "pyrocko"])
async def test_travel_time_module(
    octree: Octree,
    stations: StationInventory,
    plot: bool,
    implementation: FMMImplementation,
) -> None:
    station_list = StationList.from_inventory(stations)
    models = [
        constant_earth_model(),
        LayeredEarthModel1D(),
    ]

    for model in models:
        fmm_tracer = FastMarchingTracer(
            velocity_model=model,
            nthreads=0,
            implementation=implementation,
            interpolation_method="linear",
        )

        cake_tracer = CakeTracer(
            phases={
                "cake:P": Timing(definition="P,p"),
                "cake:S": Timing(definition="S,s"),
            },
            earthmodel=model,
        )

        await fmm_tracer.prepare(octree, stations)
        await cake_tracer.prepare(octree, stations)

        nodes = choices(octree.nodes, k=25)
        for node in nodes:
            node.split()

        fmm_times = await fmm_tracer.get_travel_times("fm:P", octree, station_list)
        cake_times = await cake_tracer.get_travel_times("cake:P", octree, station_list)

        nan_mask = np.isnan(fmm_times) | np.isnan(cake_times)

        np.testing.assert_allclose(
            fmm_times[~nan_mask],
            cake_times[~nan_mask],
            rtol=1e-1,
            atol=1e-1,
        )

        octree = octree.reset()

        for _ in range(10):
            station_selection = choices(stations.stations, k=stations.n_stations)

            fmm_times = await fmm_tracer.get_travel_times(
                "fm:P", octree, station_selection
            )
            cake_times = await cake_tracer.get_travel_times(
                "cake:P", octree, station_selection
            )

            np.testing.assert_allclose(
                fmm_times,
                cake_times,
                rtol=1e-1,
                atol=1e-1,
            )

        if plot:
            import matplotlib.pyplot as plt

            octree = octree.reset()
            fmm_times = await fmm_tracer.get_travel_times("fm:P", octree, stations)
            cake_times = await cake_tracer.get_travel_times("cake:P", octree, stations)

            for i_station, station in enumerate(stations):
                n_east = int(octree.east_bounds.width() // octree.root_node_size)
                n_north = int(octree.north_bounds.width() // octree.root_node_size)
                n_depth = int(octree.depth_bounds.width() // octree.root_node_size)

                fmm_times_station = fmm_times[:, i_station].reshape(
                    n_depth, n_north, n_east
                )
                cake_times_station = cake_times[:, i_station].reshape(
                    n_depth, n_north, n_east
                )
                tt_difference = fmm_times_station[2, :, :] - cake_times_station[2, :, :]

                data_max = np.max(np.abs(tt_difference))

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                cmap = ax.imshow(
                    fmm_times_station[2, :, :] - cake_times_station[2, :, :],
                    extent=(
                        octree.east_bounds.start,
                        octree.east_bounds.end,
                        octree.north_bounds.start,
                        octree.north_bounds.end,
                    ),
                    cmap="seismic",
                    vmin=-data_max,
                    vmax=data_max,
                )
                fig.colorbar(cmap, label="Time difference (s)")
                ax.set_title(
                    f"Station {round(station.effective_depth)} m - "
                    f"[FMM ({implementation}) - Cake difference]"
                )
                ax.set_xlabel("East (km)")
                ax.set_ylabel("North (km)")
                ax.xaxis.set_major_formatter(lambda x, pos: f"{x / KM:.1f}")
                ax.yaxis.set_major_formatter(lambda x, pos: f"{x / KM:.1f}")
                plt.show()


def test_surface_distances(octree, stations) -> None:
    station_list = StationList.from_inventory(stations)
    distances_full = surface_distances(octree.nodes, station_list)

    distances_short = surface_distances_reference(
        octree.nodes, stations, octree.location
    )
    np.testing.assert_allclose(distances_full, distances_short, rtol=1e-2)

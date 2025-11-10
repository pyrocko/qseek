from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qseek.models.layered_model import Layer, LayeredModel
from qseek.models.location import Location
from qseek.models.station import Station, Stations
from qseek.octree import Octree
from qseek.tracers.cake import CakeTracer, Timing
from qseek.tracers.fast_marching import FastMarchingTracer, StationTravelTimeTable
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


@pytest.fixture(scope="session")
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
def stations() -> Stations:
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
    return Stations(stations=stations)


@pytest.mark.asyncio
async def test_station_travel_time_table_constant(plot: bool):
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
    await table.calculate()

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
        atol=1e-2,
    )

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
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
async def test_travel_time_module(
    octree: Octree,
    stations: Stations,
    plot: bool,
) -> None:
    models = [
        constant_earth_model(),
        LayeredEarthModel1D(),
    ]

    for model in models:
        fmm_tracer = FastMarchingTracer(
            velocity_model=model,
            nthreads=0,
            implementation="scikit-fmm",
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

        fmm_times = await fmm_tracer.get_travel_times("fm:P", octree, stations)
        cake_times = await cake_tracer.get_travel_times("cake:P", octree, stations)

        if plot:
            import matplotlib.pyplot as plt

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

                data = fmm_times_station[2, :, :] - cake_times_station[2, :, :]
                data_max = np.max(np.abs(data))

                cmap = plt.imshow(
                    fmm_times_station[2, :, :] - cake_times_station[2, :, :],
                    cmap="seismic",
                    vmin=-data_max,
                    vmax=data_max,
                )
                plt.colorbar(cmap, label="Time difference (s)")
                plt.title(
                    f"Station {round(station.effective_depth)} m - "
                    "[FMM - Cake difference]"
                )
                plt.xlabel("East")
                plt.ylabel("North")
                plt.show()

        np.testing.assert_allclose(fmm_times, cake_times, rtol=1e-1)


def test_surface_distances(octree, stations):
    distances_full = surface_distances(octree.nodes, stations)

    distances_short = surface_distances_reference(
        octree.nodes, stations, octree.location
    )
    np.testing.assert_allclose(distances_full, distances_short, rtol=1e-2)

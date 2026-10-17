from __future__ import annotations

import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from random import choices

import numpy as np
import pytest

from qseek.models.layered_model import Layer, LayeredModel
from qseek.models.location import Location
from qseek.models.station import Station, StationInventory, StationList
from qseek.octree import Node, Octree
from qseek.tracers.cake import CakeTracer, Timing
from qseek.tracers.fast_marching import (
    FastMarchingTracer,
    FMMImplementation,
    StationTravelTimeTable,
)
from qseek.tracers.utils import LayeredEarthModel1D
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
            lat=50.0,
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
    """Stations with their own coordinates, like stations from a StationXML."""
    rng = np.random.default_rng(1232)
    n_stations = 5
    stations: list[Station] = []
    for i_sta in range(n_stations):
        station = Station(
            network="XX",
            station="STA%02d" % i_sta,
            lat=50.0 + rng.uniform(-0.1, 0.1),
            lon=10.0 + rng.uniform(-0.15, 0.15),
            elevation=rng.uniform(0, 1) * KM,
            depth=rng.uniform(0, 0.2) * KM,
        )
        stations.append(station)
    return StationInventory(stations=stations)


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", ["scikit-fmm", "pyrocko", "eikonalfm"])
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

    rng = np.random.default_rng(0)
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

    atol = {
        "scikit-fmm": 1e-2,
        "pyrocko": 1e-1,
        "eikonalfm": 1e-4,
    }

    np.testing.assert_allclose(
        table._travel_times,
        analytical_vp_tt,
        atol=atol[implementation],
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


def straight_ray_times(
    nodes: list[Node], stations: list[Station], velocity: float
) -> np.ndarray:
    """Travel times in a homogeneous medium, shape (n_nodes, n_stations)."""
    times = np.empty((len(nodes), len(stations)))
    for i_node, node in enumerate(nodes):
        source = node.as_location()
        for i_sta, station in enumerate(stations):
            distance = math.hypot(
                station.surface_distance_to(source),
                source.effective_depth - station.effective_depth,
            )
            times[i_node, i_sta] = distance / velocity
    return times


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", ["scikit-fmm", "pyrocko", "eikonalfm"])
async def test_travel_time_module_constant(
    octree: Octree,
    stations: StationInventory,
    plot: bool,
    implementation: FMMImplementation,
) -> None:
    station_list = StationList.from_inventory(stations)
    fmm_tracer = FastMarchingTracer(
        velocity_model=constant_earth_model(),
        implementation=implementation,
    )
    await fmm_tracer.prepare(octree, stations)

    for node in choices(octree.nodes, k=25):
        node.split()
    nodes = list(octree)

    atol = {
        "scikit-fmm": 0.02,
        "pyrocko": 0.1,
        "eikonalfm": 0.005,
    }[implementation]

    fmm_times = await fmm_tracer.get_travel_times("fm:P", nodes, station_list)
    analytical_times = straight_ray_times(nodes, list(station_list), 5.0 * KM)
    np.testing.assert_allclose(fmm_times, analytical_times, rtol=0.0, atol=atol)

    for _ in range(10):
        station_selection = choices(stations.stations, k=stations.n_stations)
        station_indices = station_list.get_indices(station_selection)
        fmm_times_selection = await fmm_tracer.get_travel_times(
            "fm:P", nodes, station_selection
        )
        np.testing.assert_allclose(
            fmm_times_selection, fmm_times[:, station_indices], rtol=0.0, atol=1e-6
        )

    # Travel times to a location come from the same tables as the node LUT
    for node in choices(nodes, k=20):
        source = node.as_location()
        times_location = fmm_tracer.get_travel_times_locations(
            "fm:P", source, list(station_list)
        )
        times_node = await fmm_tracer.get_travel_times("fm:P", [node], station_list)
        np.testing.assert_allclose(times_location, times_node[0], atol=1e-4)

    outside = Location(lat=50.0, lon=10.0, east_shift=100 * KM)
    arrivals = fmm_tracer.get_arrivals(
        "fm:P",
        event_time=datetime.now(tz=timezone.utc),
        source=outside,
        receivers=list(station_list),
    )
    assert arrivals == [None] * station_list.n_stations

    if plot:
        import matplotlib.pyplot as plt

        octree = octree.reset()
        nodes = list(octree)
        fmm_times = await fmm_tracer.get_travel_times("fm:P", nodes, station_list)
        analytical_times = straight_ray_times(nodes, list(station_list), 5.0 * KM)

        n_east = int(octree.east_bounds.width() // octree.root_node_size)
        n_north = int(octree.north_bounds.width() // octree.root_node_size)
        n_depth = int(octree.depth_bounds.width() // octree.root_node_size)
        for i_station, station in enumerate(station_list):
            tt_difference = (
                fmm_times[:, i_station] - analytical_times[:, i_station]
            ).reshape(n_depth, n_north, n_east)[2]
            data_max = np.max(np.abs(tt_difference))

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            cmap = ax.imshow(
                tt_difference,
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
                f"[FMM ({implementation}) - analytical difference]"
            )
            ax.set_xlabel("East (km)")
            ax.set_ylabel("North (km)")
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x / KM:.1f}")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x / KM:.1f}")
            plt.show()


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", ["scikit-fmm", "pyrocko", "eikonalfm"])
async def test_travel_time_module_layered(
    octree: Octree,
    stations: StationInventory,
    implementation: FMMImplementation,
) -> None:
    station_list = StationList.from_inventory(stations)
    model = LayeredEarthModel1D()
    fmm_tracer = FastMarchingTracer(
        velocity_model=model,
        implementation=implementation,
    )
    cake_tracer = CakeTracer(
        phases={"cake:P": Timing(definition="P,p")},
        earthmodel=model,
    )
    await fmm_tracer.prepare(octree, stations)
    await cake_tracer.prepare(octree, stations)

    nodes = list(octree)
    fmm_times = await fmm_tracer.get_travel_times("fm:P", nodes, station_list)
    cake_times = await cake_tracer.get_travel_times("cake:P", nodes, station_list)

    # Cake has no head waves; within this octree they are rarely the first arrival
    atol = {
        "scikit-fmm": 0.04,
        "pyrocko": 0.15,
        "eikonalfm": 0.03,
    }[implementation]
    np.testing.assert_allclose(fmm_times, cake_times, rtol=0.0, atol=atol)


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", ["scikit-fmm", "pyrocko", "eikonalfm"])
async def test_station_travel_time_table_zero_velocity(
    implementation: FMMImplementation,
) -> None:
    model = LayeredModel(
        layers=[
            Layer(top_depth=0.0, vp=1.5 * KM, vs=0.0),
            Layer(top_depth=1 * KM, vp=5.0 * KM, vs=3.0 * KM),
        ]
    )
    station = Station(station="TST", network="TT", lat=0.0, lon=0.0)
    table = StationTravelTimeTable(
        station=station,
        phase="fm:S",
        distance_max=10 * KM,
        depth_range=Range(-500.0, 5 * KM),
        grid_spacing=100.0,
        earth_model=model,
    )
    with pytest.raises(ValueError, match="fm:S velocity is zero or negative"):
        await table.calculate(implementation=implementation)

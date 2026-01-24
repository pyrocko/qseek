import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyrocko import orthodrome as od

from qseek.distance_weights import (
    _get_interstation_distances,
    _nn_median,
    _station_density,
    _station_weights,
    distance_weights,
    weights_gaussian,
)
from qseek.models.location import Location, get_coordinates

KM = 1e3


def get_distances(nodes: list[Location], stations: list[Location]) -> np.ndarray:
    node_coords = get_coordinates(nodes)
    node_coords_ecef = np.array(od.geodetic_to_ecef(*node_coords.T), dtype=np.float32).T

    station_coords = get_coordinates(stations)
    station_coords_ecef = np.array(
        od.geodetic_to_ecef(*station_coords.T), dtype=np.float32
    ).T
    return np.linalg.norm(station_coords_ecef - node_coords_ecef[:, np.newaxis], axis=2)


def sample_locations(
    n_stations: int = 20,
    bounds: float = 10 * KM,
    center: tuple[float, float] = (0.0, 0.0),
) -> list[Location]:
    rng = np.random.default_rng(1234)
    locations: list[Location] = []
    for _ in range(n_stations):
        north_shift = rng.uniform(-bounds + center[0], bounds + center[1])
        east_shift = rng.uniform(-bounds + center[0], bounds + center[1])
        loc = Location(
            lat=0.0,
            lon=0.0,
            north_shift=north_shift,
            east_shift=east_shift,
        )
        locations.append(loc)
    return locations


@pytest.mark.plot
def test_station_density():
    stations = sample_locations(n_stations=20)
    cluster = sample_locations(
        n_stations=10,
        bounds=1 * KM,
        center=(5 * KM, 5 * KM),
    )

    stations.extend(cluster)

    distances = _get_interstation_distances(stations)
    nn_median = _nn_median(distances)
    densities = _station_density(distances, radius=nn_median)
    weights = 1 - (densities - densities.min()) / densities.max()
    weights_inv = 1.0 / np.sqrt(densities)

    colors = plt.get_cmap("turbo_r")(weights)

    fig = plt.figure()
    ax_density = fig.add_subplot(221)
    ax_density.set_title("Station Density")
    ax_density.scatter(
        [sta.east_shift for sta in stations],
        [sta.north_shift for sta in stations],
        s=densities * 100,
        fc=colors,
        ec="none",
        alpha=0.7,
    )
    ax_density.scatter(
        [sta.east_shift for sta in stations],
        [sta.north_shift for sta in stations],
        s=nn_median,
        edgecolors="black",
        facecolors="none",
        alpha=0.1,
    )
    ax_density.text(
        0.05,
        0.05,
        f"NN median: {nn_median / KM:.2f} km",
        transform=ax_density.transAxes,
        ha="left",
        va="bottom",
    )

    ax_weight = fig.add_subplot(222)
    ax_weight.set_title("Station Weights")
    ax_weight.scatter(
        [loc.east_shift for loc in stations],
        [loc.north_shift for loc in stations],
        s=weights * 200,
        ec="none",
        fc=colors,
        alpha=0.7,
    )

    ax_sta_density = fig.add_subplot(223)
    densities_sort = np.argsort(densities)
    ax_sta_density.scatter(
        np.arange(len(densities)),
        densities[densities_sort],
        fc=colors[densities_sort],
        ec="none",
    )
    ax_sta_density.set_xlabel("Station Index (sorted)")
    ax_sta_density.set_ylabel("Density")
    ax_sta_density.set_ylim(0, ax_sta_density.get_ylim()[1])

    ax_sta_weights = fig.add_subplot(224)
    ax_sta_weights.scatter(
        np.arange(len(weights)),
        weights[densities_sort],
        fc=colors[densities_sort],
        ec="none",
    )
    ax_sta_weights.scatter(
        np.arange(len(weights)),
        weights_inv[densities_sort],
        fc=colors[densities_sort],
        ec="none",
        alpha=0.4,
    )
    ax_sta_weights.set_xlabel("Station Index (sorted)")
    ax_sta_weights.set_ylabel("Weight")
    ax_sta_weights.set_ylim(0, 1.05)

    ax_density.set_aspect("equal")
    ax_weight.set_aspect("equal")
    for ax in [ax_density, ax_weight]:
        ax.set_xlabel("East Shift (km)")
        ax.set_ylabel("North Shift (km)")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / KM:.0f}")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y / KM:.0f}")
    for ax in fig.axes:
        ax.grid(alpha=0.3)

    plt.show()


@pytest.mark.plot
def test_distance_weights():
    stations = sample_locations(n_stations=20)
    cluster = sample_locations(
        n_stations=10,
        bounds=1 * KM,
        center=(5 * KM, 5 * KM),
    )

    stations.extend(cluster)

    # source nodes
    nodes = [
        Location(lat=0.0, lon=0.0, north_shift=-5000.0, east_shift=0.0, depth=1 * KM),
        Location(lat=0.0, lon=0.0, north_shift=-5000.0, east_shift=0.0, depth=6 * KM),
    ]
    station_weight_plateau = 4

    interstation_distances = _get_interstation_distances(stations)
    distances = get_distances(nodes, stations)

    station_weight = _station_weights(interstation_distances)
    distance_weight = distance_weights(
        distances,
        station_weight,
        station_weight_plateau=station_weight_plateau,
        station_weight_taper=4 * station_weight_plateau,
    )
    total_weight = distance_weight * station_weight[np.newaxis, :]

    distance_sort = np.argsort(distances, axis=1)
    distances_sorted = np.take_along_axis(distances, distance_sort, axis=1)
    station_weight_sorted = station_weight[distance_sort]

    node = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(
        distances[node],
        distance_weight[node],
        label="Distance Weight",
    )

    ax.scatter(
        distances[node],
        total_weight[node],
        c="red",
        alpha=0.7,
        label="Total Weight (Distance x Station)",
    )

    ax_station_weight = ax.twinx()

    ax_station_weight.stairs(
        np.cumsum(station_weight_sorted[node]),
        [*distances_sorted[node], distances_sorted[node, -1] + 10 * KM],
        ls="-",
        alpha=0.7,
        label="Cumulative Station Weight",
    )

    ax.set_xlim(0, distances[node].max())
    ax.plot()
    ax.grid(alpha=0.3)
    ax.set_xlabel("Station Distance (km)")
    ax.set_ylabel("Station Weight")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / KM:.0f}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.show()


def test_nearest_neighbor_median():
    locations = [
        Location(lat=0.0, lon=0.0),
        Location(lat=0.0, lon=0.1),
        Location(lat=0.1, lon=0.0),
        Location(lat=0.1, lon=0.1),
    ]
    distances = get_distances(locations, locations)
    np.fill_diagonal(distances, np.nan)

    median_distance = _nn_median(distances)
    expected_distance = locations[0].distance_to(locations[1])

    np.testing.assert_allclose(median_distance, expected_distance, rtol=1e-2)


@pytest.mark.plot
def test_weights_gaussian_min_stations():
    rng = np.random.default_rng(1234)
    n_nodes = 10
    n_stations = 30

    station_distances = rng.uniform(0, 50 * KM, size=(n_nodes, n_stations))

    n_stations_plateau = 4
    n_stations_taper = 6

    station_weights_taper = weights_gaussian(
        station_distances,
        n_stations_plateau=n_stations_plateau,
        n_stations_taper_distance=n_stations_taper,
        taper_distance=10 * KM,
        waterlevel=0.1,
    )

    node = 1

    sorted_distances = np.argsort(station_distances[node])
    distances = station_distances[node][sorted_distances]
    weights_taper = station_weights_taper[node][sorted_distances]

    dist_cutoff = distances[n_stations_plateau - 1]
    # n_stations_taper = min(n_stations_taper + n_stations_plateau, distances.size)
    distance_taper = distances[n_stations_taper + n_stations_plateau - 1]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.scatter(
        distances,
        weights_taper,
        s=30,
        ec="none",
        label="Station",
    )

    ax.axvline(
        dist_cutoff,
        color="red",
        linestyle="--",
        alpha=0.8,
        zorder=-1,
    )
    ax.axvline(
        distance_taper,
        color="green",
        linestyle="--",
        alpha=0.8,
        zorder=-1,
    )

    ax.hlines(
        0.1,
        distance_taper,
        50 * KM,
        color="black",
        linestyle="--",
        zorder=-1,
        alpha=0.8,
    )

    ax.text(
        dist_cutoff - 0.5 * KM,
        0.95,
        f"$N$ stations\nplateau = {n_stations_plateau}",
        c="black",
        ha="right",
        va="top",
        # rotation=70.0,
    )
    ax.text(
        distance_taper - 0.5 * KM,
        weights_taper[n_stations_taper + n_stations_plateau - 1] - 0.05,
        f"$N$ stations\ntaper = {n_stations_taper}",
        c="black",
        ha="right",
        va="top",
        # rotation=70.0,
    )
    ax.text(
        48 * KM,
        0.03,
        "Waterlevel = 0.1",
        c="black",
        ha="right",
    )

    ax.set_xlabel("Station Distance (km)")
    ax.set_ylabel("Station Weight")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / KM:.0f}")
    ax.grid(alpha=0.3)

    ax.set_xlim(0, 50 * KM)
    ax.set_ylim(0, 1.05)

    twin_ax = ax.twinx()
    twin_ax.set_ylim(0, 105)
    twin_ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}")
    twin_ax.grid(False)

    twin_ax.stairs(
        (weights_taper.cumsum() / weights_taper.sum()) * 100,
        [*distances, distances[-1] + 10 * KM],
        color="black",
        alpha=0.8,
        label="Cumulative Weight (%)",
    )
    twin_ax.set_ylabel("Cumulative Weight (%)")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    test_weights_gaussian_min_stations()

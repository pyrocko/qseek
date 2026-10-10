from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qseek.models.location import Location
from qseek.models.station import Station, StationInventory
from qseek.octree import Octree
from qseek.station_weights import (
    NEAREST_NEIGHBOR_TAPER,
    DistanceWeights,
    LogLogisticWeights,
    StationDensityWeights,
    colocated_sensors,
    independent_stations_distance,
    interstation_distances,
    nearest_neighbor_distance,
    station_density,
    station_independence,
    weights_gaussian,
    weights_log_logistic,
    weights_plateau_gaussian,
)

KM = 1e3


def local_locations(positions_km: list[tuple[float, float]]) -> list[Location]:
    return [
        Location(lat=0.0, lon=0.0, north_shift=north * KM, east_shift=east * KM)
        for north, east in positions_km
    ]


def local_stations(positions_km: list[tuple[float, float]]) -> StationInventory:
    return StationInventory(
        stations=[
            Station(
                network="XX",
                station=f"S{idx:02d}",
                lat=10.0,
                lon=10.0,
                north_shift=north * KM,
                east_shift=east * KM,
            )
            for idx, (north, east) in enumerate(positions_km)
        ]
    )


def test_weights_gaussian():
    distance_taper = 10 * KM
    # Two nodes, distances to five stations
    distances = (
        np.array(
            [
                [1.0, 2.0, 3.0, 8.0, 13.0],
                [5.0, 1.0, 20.0, 3.0, 50.0],
            ]
        )
        * KM
    )
    weights = weights_gaussian(distances, distance_taper, required_stations=3)

    assert weights.shape == distances.shape
    # The closest stations have full weight
    np.testing.assert_equal(weights[0, :3], 1.0)
    np.testing.assert_equal(weights[1, [0, 1, 3]], 1.0)
    # The taper is the full width at half maximum beyond the closest stations
    assert weights[0, 3] == pytest.approx(0.5, abs=0.01)
    # Weights decrease with distance
    assert weights[0, 3] > weights[0, 4] > 0.0
    assert weights[1, 2] > weights[1, 4]


def test_weights_gaussian_waterlevel():
    rng = np.random.default_rng(42)
    distances = rng.uniform(0, 100 * KM, size=(10, 20))
    weights = weights_gaussian(distances, 5 * KM, required_stations=2, waterlevel=0.1)
    assert weights.min() >= 0.1
    assert weights.max() == pytest.approx(1.0)


def test_weights_gaussian_required_stations():
    distances = np.array([[1.0, 50.0, 100.0]]) * KM
    # More required stations than available, all stations get full weight
    np.testing.assert_equal(
        weights_gaussian(distances, 1 * KM, required_stations=10), 1.0
    )
    with pytest.raises(ValueError):
        weights_gaussian(distances, 1 * KM, required_stations=0)


@pytest.mark.plot
def test_weights_gaussian_min_stations():
    rng = np.random.default_rng(42)
    n_nodes = 10
    n_stations = 50

    station_distances = rng.uniform(0, 50 * KM, size=(n_nodes, n_stations))

    required_stations = 5
    distance_taper = 20 * KM

    station_weights = weights_gaussian(
        station_distances,
        distance_taper=distance_taper,
        required_stations=required_stations,
        waterlevel=0.1,
    )

    node = 1

    sorted_distances = np.argsort(station_distances[node])
    distances = station_distances[node][sorted_distances]
    weights = station_weights[node][sorted_distances]
    dist_cutoff = distances[required_stations - 1]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.scatter(
        distances,
        weights,
        alpha=0.9,
        s=30,
        ec="none",
        label="Station",
    )

    ax.axvline(dist_cutoff, color="red", linestyle="--", alpha=0.8, zorder=-1)
    ax.axvline(
        dist_cutoff + distance_taper,
        color="green",
        linestyle="--",
        alpha=0.8,
        zorder=-1,
    )

    ax.hlines(
        0.1,
        dist_cutoff + distance_taper,
        50 * KM,
        color="black",
        linestyle="--",
        zorder=-1,
        alpha=0.8,
    )

    ax.text(
        dist_cutoff * 0.8,
        0.3,
        f"Req. closest stations = {required_stations}",
        c="black",
        ha="right",
        rotation=90.0,
    )
    ax.text(
        dist_cutoff + distance_taper / 2,
        0.03,
        f"Distance taper ={distance_taper / KM:.0f} km",
        c="black",
        ha="center",
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

    twin_ax.plot(
        distances,
        weights.cumsum() * 100 / weights.sum(),
        color="black",
        alpha=0.8,
        label="Cumulative Weight (%)",
    )
    twin_ax.set_ylabel("Cumulative Weight (%)")
    fig.tight_layout()

    plt.show()


def test_interstation_distances():
    distances = interstation_distances(
        local_locations([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)])
    )
    assert distances.shape == (3, 3)
    assert np.isnan(np.diag(distances)).all()
    np.testing.assert_allclose(distances, distances.T)
    # Shifts are converted on a sphere, distances on the ellipsoid
    np.testing.assert_allclose(distances[0, 1], 3 * KM, rtol=1e-2)
    np.testing.assert_allclose(distances[1, 2], 5 * KM, rtol=1e-2)


def test_nearest_neighbor_distance():
    sites = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)]
    distances = interstation_distances(local_locations(sites))
    assert nearest_neighbor_distance(distances) == pytest.approx(10 * KM, rel=1e-2)

    # Co-located sensors are one site
    colocated = interstation_distances(local_locations(sites + sites[:3]))
    assert nearest_neighbor_distance(colocated) == pytest.approx(
        nearest_neighbor_distance(distances)
    )
    # A single site has no neighbors
    single = interstation_distances(local_locations([(0.0, 0.0), (0.0, 0.0)]))
    assert np.isnan(nearest_neighbor_distance(single))
    np.testing.assert_equal(station_density(single), 1.0)


def test_station_density():
    distances = interstation_distances(
        local_locations([(0.0, 0.0), (1.0, 0.0), (10.0, 0.0)])
    )
    # A wide kernel sees more neighbors than a narrow one
    assert np.all(
        station_density(distances, radius=5 * KM)
        > station_density(distances, radius=0.1 * KM)
    )
    # Without neighbors in the kernel, the density is the station itself
    np.testing.assert_allclose(station_density(distances, radius=1.0), 1.0)


def test_station_independence_equidistant():
    # Stations on a circle have the same density
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    independence = station_independence(
        interstation_distances(
            local_locations([(np.cos(a), np.sin(a)) for a in angles])
        )
    )
    np.testing.assert_allclose(independence, 1.0, rtol=2e-3)


def test_station_independence_cluster():
    # A tight pair, 100 m apart, and two isolated stations
    independence = station_independence(
        interstation_distances(
            local_locations([(0.0, 0.0), (0.1, 0.0), (20.0, 0.0), (0.0, 20.0)])
        )
    )
    np.testing.assert_allclose(independence[2:], 1.0, atol=1e-2)
    assert independence[0] == pytest.approx(independence[1], rel=1e-2)
    assert np.all(independence[:2] < 0.6)


def test_station_independence_colocated():
    # Co-located sensors of one site, e.g. location codes 00 and 10
    rng = np.random.default_rng(0)
    sites = [tuple(rng.uniform(-10, 10, size=2)) for _ in range(10)]
    distances = interstation_distances(local_locations(sites + sites[:6]))
    np.testing.assert_equal(colocated_sensors(distances), [2] * 6 + [1] * 4 + [2] * 6)

    # The sensors of a site share the independent station count of the site
    independence = station_independence(distances)
    independence_sites = station_independence(
        interstation_distances(local_locations(sites))
    )
    np.testing.assert_allclose(independence[:6], independence_sites[:6] / 2)
    np.testing.assert_allclose(independence[:6], independence[10:])
    np.testing.assert_allclose(independence[6:10], independence_sites[6:10])
    assert independence.sum() == pytest.approx(independence_sites.sum())

    # Two sensors of a site count as the site alone
    three_sites = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)]
    pair = station_independence(
        interstation_distances(local_locations(three_sites[:1] + three_sites))
    )
    single = station_independence(interstation_distances(local_locations(three_sites)))
    np.testing.assert_allclose(pair[:2], single[0] / 2)
    np.testing.assert_allclose(pair[2:], single[1:])


def test_station_independence_range():
    rng = np.random.default_rng(0)
    for n_stations in (2, 5, 30):
        independence = station_independence(
            interstation_distances(
                local_locations(
                    [tuple(rng.uniform(-10, 10, size=2)) for _ in range(n_stations)]
                )
            )
        )
        assert independence.shape == (n_stations,)
        assert np.all((independence > 0.0) & (independence <= 1.0))
        assert independence.max() == pytest.approx(1.0)


def test_independent_stations_distance():
    distances = np.arange(1, 13, dtype=float)[np.newaxis, :] * KM
    # Stations that count as half: 4 independent stations need 8 of them
    independence = np.full(12, 0.5)
    np.testing.assert_equal(
        independent_stations_distance(distances, independence, 4.0), [[8 * KM]]
    )
    # The network has fewer independent stations: the most distant station
    np.testing.assert_equal(
        independent_stations_distance(distances, independence, 20.0), [[12 * KM]]
    )
    # The order of the stations does not matter
    perm = np.random.default_rng(2).permutation(12)
    np.testing.assert_equal(
        independent_stations_distance(distances[:, perm], independence[perm], 4.0),
        [[8 * KM]],
    )


def test_weights_plateau_gaussian():
    distances = np.arange(1, 21, dtype=float)[np.newaxis, :] * KM
    weights = weights_plateau_gaussian(
        distances,
        np.ones(20),
        plateau_stations=4.0,
        taper_stations=12.0,
        max_taper_ratio=None,
    )
    # Gaussian centered at the plateau distance, sigma half the taper distance
    plateau, sigma = 4 * KM, 12 * KM / 2
    expected = np.exp(-((distances - plateau) ** 2) / (2 * sigma**2))
    expected[distances <= plateau] = 1.0
    np.testing.assert_allclose(weights, expected, rtol=1e-6)

    # Isolated stations: the plateau holds the N closest stations
    rng = np.random.default_rng(1)
    distances = rng.uniform(0, 50 * KM, size=(5, 20))
    weights = weights_plateau_gaussian(
        distances, np.ones(20), plateau_stations=4.0, taper_stations=12.0
    )
    sorted_weights = np.take_along_axis(weights, np.argsort(distances, axis=1), axis=1)
    np.testing.assert_equal(sorted_weights[:, :4], 1.0)
    assert np.all(sorted_weights[:, 4] < 1.0)
    assert np.all(np.diff(sorted_weights[:, 4:], axis=1) <= 0.0)


def test_weights_plateau_gaussian_max_taper_ratio():
    # A gap: three close stations, then distant stations
    distances = np.array([[1.0, 2.0, 3.0, 50.0, 60.0, 70.0, 80.0]]) * KM
    plateau, taper = 3 * KM, 70 * KM
    uncapped = weights_plateau_gaussian(
        distances, np.ones(7), 3.0, 6.0, max_taper_ratio=None
    )
    capped = weights_plateau_gaussian(distances, np.ones(7), 3.0, 6.0)
    # The cap limits sigma to the plateau distance
    for weights, sigma in ((uncapped, taper / 2), (capped, plateau)):
        expected = np.exp(-((distances - plateau) ** 2) / (2 * sigma**2))
        expected[distances <= plateau] = 1.0
        np.testing.assert_allclose(weights, expected, rtol=1e-6)
    assert np.all(capped[0, 3:] < uncapped[0, 3:])

    # Without a gap the cap does not bind
    distances = np.arange(1, 21, dtype=float)[np.newaxis, :] * KM
    np.testing.assert_allclose(
        weights_plateau_gaussian(distances, np.ones(20), 4.0, 8.0),
        weights_plateau_gaussian(distances, np.ones(20), 4.0, 8.0, None),
    )


def test_weights_log_logistic():
    plateau = np.array([[2.0], [20.0]]) * KM
    distances = np.array([[2.0, 4.4, 8.8, 100.0], [20.0, 44.0, 88.0, 1000.0]]) * KM
    weights = weights_log_logistic(distances, plateau, taper_scale=2.2)
    # Half weight at taper_scale times the plateau distance
    np.testing.assert_allclose(weights[:, 1], 0.5)
    # Scale free: the same weights for a network ten times larger
    np.testing.assert_allclose(weights[0], weights[1])
    assert np.all(np.diff(weights, axis=1) < 0.0)
    assert weights[0, 0] == pytest.approx(1 / (1 + (1 / 2.2) ** 4))
    # A steeper taper decays faster beyond the half weight distance
    steep = weights_log_logistic(distances, plateau, taper_scale=2.2, taper_exponent=8)
    assert np.all(steep[:, 2:] < weights[:, 2:])


@pytest.fixture
def clustered_stations() -> StationInventory:
    # Sparse network with a dense cluster
    rng = np.random.default_rng(3)
    sparse = [tuple(rng.uniform(-30, 30, size=2)) for _ in range(12)]
    cluster = [tuple(5 + rng.normal(0, 0.3, size=2)) for _ in range(8)]
    return local_stations(sparse + cluster)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "weights_model",
    [
        DistanceWeights(),
        DistanceWeights(distance_taper="nearest_neighbor"),
        DistanceWeights(distance_taper=5 * KM, waterlevel=0.1),
        StationDensityWeights(),
        LogLogisticWeights(),
    ],
)
async def test_station_weights_model(
    weights_model, octree: Octree, clustered_stations: StationInventory
):
    weights_model = weights_model.model_copy(deep=True)
    weights_model.prepare(clustered_stations, octree)
    stations = list(clustered_stations)

    nodes = octree.nodes[:50]
    weights = await weights_model.get_weights(nodes, stations)
    assert weights.shape == (len(nodes), len(stations))
    assert np.all((weights > 0.0) & (weights <= 1.0))
    assert weights.min() >= weights_model.waterlevel

    # Weights follow the order of the stations
    perm = np.random.default_rng(4).permutation(len(stations))
    weights_perm = await weights_model.get_weights(nodes, [stations[i] for i in perm])
    np.testing.assert_allclose(weights_perm, weights[:, perm], rtol=1e-5)

    # Unavailable stations: weights for the remaining stations
    subset = stations[::2]
    weights_subset = await weights_model.get_weights(nodes, subset)
    assert weights_subset.shape == (len(nodes), len(subset))

    # Child nodes are not in the lookup table yet
    children = list(octree.nodes[0].split())
    weights_children = await weights_model.get_weights(children, stations)
    assert weights_children.shape == (len(children), len(stations))


@pytest.mark.asyncio
@pytest.mark.parametrize("distance_taper", ["nearest_neighbor", "mean_interstation"])
async def test_distance_weights_taper(
    distance_taper, octree: Octree, clustered_stations: StationInventory
):
    weights_model = DistanceWeights(distance_taper=distance_taper)
    weights_model.prepare(clustered_stations, octree)
    distances = interstation_distances(list(clustered_stations))
    if distance_taper == "nearest_neighbor":
        expected_taper = NEAREST_NEIGHBOR_TAPER * nearest_neighbor_distance(distances)
    else:
        expected_taper = 2 * clustered_stations.mean_interstation_distance()
    assert weights_model.distance_taper == pytest.approx(expected_taper)

    nodes = octree.nodes[:20]
    weights = await weights_model.get_weights(nodes, list(clustered_stations))
    expected = weights_gaussian(
        weights_model.get_distances(nodes), expected_taper, required_stations=4
    )
    np.testing.assert_allclose(weights, expected, rtol=1e-6)


@pytest.mark.asyncio
async def test_station_density_weights_subset(
    octree: Octree, clustered_stations: StationInventory
):
    weights_model = StationDensityWeights()
    weights_model.prepare(clustered_stations, octree)
    stations = list(clustered_stations)
    nodes = octree.nodes[:20]

    # The independent station counts adapt to the available stations
    subset = stations[::2]
    weights = await weights_model.get_weights(nodes, subset)
    expected = weights_plateau_gaussian(
        weights_model.get_distances(nodes)[:, ::2],
        station_independence(interstation_distances(subset)),
        plateau_stations=weights_model.plateau_stations,
        taper_stations=weights_model.taper_stations,
    )
    np.testing.assert_allclose(weights, expected, rtol=1e-5)

    # The cluster stations count less than the sparse stations
    independence = weights_model.get_independence(np.arange(len(stations)))
    assert independence[12:].max() < independence[:12].min()


def test_station_weights_config():
    from pydantic import ValidationError

    from qseek.search import Search

    assert isinstance(Search().station_weights, StationDensityWeights)
    assert Search.model_validate({"station_weights": None}).station_weights is None

    for model in (
        DistanceWeights(required_closest_stations=6),
        StationDensityWeights(plateau_stations=3.0, max_taper_ratio=None),
        LogLogisticWeights(),
    ):
        config = {"station_weights": model.model_dump(mode="json")}
        assert Search.model_validate(config).station_weights == model

    with pytest.raises(ValidationError):
        Search.model_validate({"station_weights": {"weights": "UnknownWeights"}})


if __name__ == "__main__":
    test_weights_gaussian_min_stations()

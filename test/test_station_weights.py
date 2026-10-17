from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qseek.models.location import Location, get_coordinates
from qseek.models.station import Station, StationInventory
from qseek.octree import Octree
from qseek.station_weights import (
    NEAREST_NEIGHBOR_TAPER,
    DensityPlateauWeights,
    DensityTaperWeights,
    DistanceWeights,
    StationDensityWeights,
    colocated_sensors,
    condition_number_radius,
    density_kernel_radius,
    independent_station_counts,
    independent_stations_distances,
    interstation_distances,
    nearest_neighbor_distance,
    station_density,
    weights_gaussian,
    weights_gaussian_density,
    weights_gaussian_independent,
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
    )
    station_weights = 0.9 * station_weights + 0.1  # waterlevel 0.1

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


def test_interstation_distances_blocks():
    # More stations than one block of rows, with co-located sensors
    from pyrocko import orthodrome as od

    from qseek.station_weights import INTERSTATION_BLOCK

    rng = np.random.default_rng(5)
    sites = [
        tuple(rng.uniform(-30, 30, size=2)) for _ in range(INTERSTATION_BLOCK + 50)
    ]
    locations = local_locations(sites + sites[:20])
    distances = interstation_distances(locations)
    assert distances.dtype == np.float32

    coords = get_coordinates(locations)
    ecef = np.array(od.geodetic_to_ecef(*coords.T)).T
    expected = np.linalg.norm(ecef - ecef[:, np.newaxis], axis=2)
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(distances, expected, rtol=1e-6, atol=1e-2)

    # The density in row blocks equals a float64 calculation in one piece
    radius = nearest_neighbor_distance(expected)
    sites_expected = np.where(expected <= 50.0, np.nan, expected)
    n_sensors = 1 + np.sum(expected <= 50.0, axis=1)
    kernel = np.exp(-(sites_expected**2) / (2 * radius**2)) / n_sensors[np.newaxis]
    np.testing.assert_allclose(
        station_density(distances), np.nansum(kernel, axis=1) + 1.0, rtol=1e-5
    )


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


def test_independent_station_counts_equidistant():
    # Stations on a circle have the same density
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    independence = independent_station_counts(
        interstation_distances(
            local_locations([(np.cos(a), np.sin(a)) for a in angles])
        )
    )
    np.testing.assert_allclose(independence, 1.0, rtol=2e-3)


def test_independent_station_counts_cluster():
    # A tight pair, 100 m apart, and two isolated stations
    independence = independent_station_counts(
        interstation_distances(
            local_locations([(0.0, 0.0), (0.1, 0.0), (20.0, 0.0), (0.0, 20.0)])
        )
    )
    np.testing.assert_allclose(independence[2:], 1.0, atol=1e-2)
    assert independence[0] == pytest.approx(independence[1], rel=1e-2)
    assert np.all(independence[:2] < 0.6)


def test_independent_station_counts_colocated():
    # Co-located sensors of one site, e.g. location codes 00 and 10
    rng = np.random.default_rng(0)
    sites = [tuple(rng.uniform(-10, 10, size=2)) for _ in range(10)]
    distances = interstation_distances(local_locations(sites + sites[:6]))
    np.testing.assert_equal(colocated_sensors(distances), [2] * 6 + [1] * 4 + [2] * 6)

    # The sensors of a site share the independent station count of the site
    independence = independent_station_counts(distances)
    independence_sites = independent_station_counts(
        interstation_distances(local_locations(sites))
    )
    np.testing.assert_allclose(independence[:6], independence_sites[:6] / 2, rtol=1e-5)
    np.testing.assert_allclose(independence[:6], independence[10:])
    np.testing.assert_allclose(independence[6:10], independence_sites[6:10], rtol=1e-5)
    assert independence.sum() == pytest.approx(independence_sites.sum(), rel=1e-5)

    # Two sensors of a site count as the site alone
    three_sites = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)]
    pair = independent_station_counts(
        interstation_distances(local_locations(three_sites[:1] + three_sites))
    )
    single = independent_station_counts(
        interstation_distances(local_locations(three_sites))
    )
    np.testing.assert_allclose(pair[:2], single[0] / 2, rtol=1e-5)
    np.testing.assert_allclose(pair[2:], single[1:], rtol=1e-5)


def test_independent_station_counts_range():
    rng = np.random.default_rng(0)
    for n_stations in (2, 5, 30):
        independence = independent_station_counts(
            interstation_distances(
                local_locations(
                    [tuple(rng.uniform(-10, 10, size=2)) for _ in range(n_stations)]
                )
            )
        )
        assert independence.shape == (n_stations,)
        assert np.all((independence > 0.0) & (independence <= 1.0))
        assert independence.max() == pytest.approx(1.0)


def test_independent_stations_distances():
    distances = np.arange(1, 13, dtype=float)[np.newaxis, :] * KM
    # Stations that count as half: 4 independent stations need 8 of them
    independence = np.full(12, 0.5)
    np.testing.assert_equal(
        independent_stations_distances(distances, independence, (4.0,))[0], [[8 * KM]]
    )
    # The network has fewer independent stations: the most distant station
    np.testing.assert_equal(
        independent_stations_distances(distances, independence, (20.0,))[0], [[12 * KM]]
    )
    # The order of the stations does not matter
    perm = np.random.default_rng(2).permutation(12)
    np.testing.assert_equal(
        independent_stations_distances(distances[:, perm], independence[perm], (4.0,))[
            0
        ],
        [[8 * KM]],
    )


def test_weights_plateau_gaussian():
    distances = np.arange(1, 21, dtype=float)[np.newaxis, :] * KM
    weights = weights_plateau_gaussian(
        distances,
        np.ones(20),
        plateau_stations=4.0,
        taper_stations=12.0,
        taper_cap=None,
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


def test_weights_plateau_gaussian_taper_cap():
    # A gap: three close stations, then distant stations
    distances = np.array([[1.0, 2.0, 3.0, 50.0, 60.0, 70.0, 80.0]]) * KM
    plateau, taper = 3 * KM, 70 * KM
    uncapped = weights_plateau_gaussian(distances, np.ones(7), 3.0, 6.0, taper_cap=None)
    capped = weights_plateau_gaussian(distances, np.ones(7), 3.0, 6.0, taper_cap=1.0)
    # The cap limits sigma to the plateau distance
    for weights, sigma in ((uncapped, taper / 2), (capped, plateau)):
        expected = np.exp(-((distances - plateau) ** 2) / (2 * sigma**2))
        expected[distances <= plateau] = 1.0
        np.testing.assert_allclose(weights, expected, rtol=1e-6)
    assert np.all(capped[0, 3:] < uncapped[0, 3:])

    # Without a gap the cap does not bind
    distances = np.arange(1, 21, dtype=float)[np.newaxis, :] * KM
    np.testing.assert_allclose(
        weights_plateau_gaussian(distances, np.ones(20), 4.0, 8.0, 1.0),
        weights_plateau_gaussian(distances, np.ones(20), 4.0, 8.0, None),
    )


@pytest.fixture
def clustered_stations() -> StationInventory:
    # Sparse network with a dense cluster
    rng = np.random.default_rng(3)
    sparse = [tuple(rng.uniform(-30, 30, size=2)) for _ in range(12)]
    cluster = [tuple(5 + rng.normal(0, 0.3, size=2)) for _ in range(8)]
    return local_stations(sparse + cluster)


def test_weights_gaussian_independent():
    rng = np.random.default_rng(5)
    distances = rng.uniform(0.5, 40.0, size=(50, 20)) * KM
    taper = 20 * KM
    # Every station an independent station: the plateau of the 4 closest stations
    np.testing.assert_allclose(
        weights_gaussian_independent(distances, np.ones(20), taper, 4.0),
        weights_gaussian(distances, taper, 4),
        rtol=1e-6,
    )
    # Stations that count less extend the plateau: the weights only grow
    counts = rng.uniform(0.2, 1.0, size=20)
    weights = weights_gaussian_independent(distances, counts, taper, 4.0)
    assert np.all(weights >= weights_gaussian(distances, taper, 4) - 1e-9)
    assert np.all(np.sort(weights, axis=1)[:, -4:] == 1.0)


def test_weights_gaussian_density():
    rng = np.random.default_rng(6)
    distances = rng.uniform(0.5, 40.0, size=(50, 20)) * KM
    taper = 20 * KM
    plain = weights_gaussian(distances, taper, 4)
    # Factors of one: the distance weights
    np.testing.assert_allclose(
        weights_gaussian_density(distances, np.ones(20), taper, 4), plain
    )
    factors = rng.uniform(0.2, 1.0, size=20)
    weights = weights_gaussian_density(distances, factors, taper, 4)
    # The closest stations keep their full weight, the others weigh less
    plateau = plain == 1.0
    assert np.all(weights[plateau] == 1.0)
    assert np.all(weights[~plateau] <= plain[~plateau])
    # Far away the weight approaches the distance weight times the factor
    far = plain < 0.01
    np.testing.assert_allclose(
        weights[far], (plain * factors[np.newaxis, :])[far], rtol=0.02
    )


def test_density_kernel_radius(clustered_stations: StationInventory):
    distances = interstation_distances(list(clustered_stations))
    assert density_kernel_radius(distances, "nearest_neighbor") == (
        nearest_neighbor_distance(distances)
    )
    assert density_kernel_radius(distances, "mean_interstation") == pytest.approx(
        np.nanmean(distances) / np.sqrt(2.0)
    )
    assert density_kernel_radius(distances, 5 * KM) == 5 * KM

    # The widest radius at which the weight ratio is a third of its peak
    radius = condition_number_radius(distances)
    squared = np.nan_to_num(distances.astype(np.float64), nan=0.0) ** 2

    def ratio(radius: float) -> float:
        weights = 1.0 / np.exp(-squared / (2 * radius**2)).sum(axis=1)
        return float(weights.max() / weights.min())

    peak = max(ratio(r) for r in np.geomspace(100.0, 1000 * KM, 400))
    assert ratio(radius) == pytest.approx(peak / 3, rel=0.1)
    assert ratio(radius * 1.2) < peak / 3

    # One site: no radius, all stations count the same
    single = interstation_distances(local_locations([(0.0, 0.0), (0.0, 0.0)]))
    assert np.isnan(condition_number_radius(single))
    np.testing.assert_allclose(
        independent_station_counts(single, "condition_number"), [0.5, 0.5]
    )


def test_independent_station_counts_radius(clustered_stations: StationInventory):
    distances = interstation_distances(list(clustered_stations))
    narrow = independent_station_counts(distances)
    np.testing.assert_array_equal(
        narrow, independent_station_counts(distances, "nearest_neighbor")
    )
    for radius in ("mean_interstation", "condition_number", 20 * KM):
        counts = independent_station_counts(distances, radius)
        assert np.all((counts > 0.0) & (counts <= 1.0))
        # The cluster counts less than the sparse stations
        assert np.median(counts[12:]) < np.median(counts[:12])
    # The condition number radius discounts the cluster less than the nearest
    # neighbor distance
    wide = independent_station_counts(distances, "condition_number")
    assert wide.max() / wide.min() < narrow.max() / narrow.min()
    assert np.median(wide[12:]) > np.median(narrow[12:])


def weights_roughness(weights: np.ndarray, step: float) -> float:
    """Largest L1 change of the normalized weights between neighboring nodes, per m."""
    weights = weights / weights.sum(axis=1, keepdims=True)
    return float(np.abs(np.diff(weights, axis=0)).sum(axis=1).max() / step)


def test_station_weights_roughness(clustered_stations: StationInventory):
    # Weights that change steeply between nodes move the maximum of the stack. The
    # plateau in independent stations keeps the weights as smooth as the distance
    # weights; the short taper of the capped density weights does not.
    locations = list(clustered_stations)
    counts = independent_station_counts(interstation_distances(locations))
    counts_wide = independent_station_counts(
        interstation_distances(locations), "condition_number"
    )
    taper = 2 * clustered_stations.mean_interstation_distance()

    step = 50.0
    sta = np.array([(s.east_shift, s.north_shift) for s in locations])
    roughness: dict[str, list[float]] = {
        "distance": [],
        "plateau": [],
        "taper": [],
        "plateau_condition": [],
        "taper_condition": [],
        "capped": [],
    }
    for angle in np.radians([0.0, 45.0, 90.0, 135.0]):
        # Profiles at 3 km depth through the network
        offsets = np.arange(-30 * KM, 30 * KM + step, step)
        line = np.stack([np.cos(angle) * offsets, np.sin(angle) * offsets], axis=1)
        horizontal = np.linalg.norm(line[:, np.newaxis] - sta[np.newaxis], axis=2)
        distances = np.hypot(horizontal, 3 * KM)

        roughness["distance"].append(
            weights_roughness(weights_gaussian(distances, taper, 4), step)
        )
        roughness["plateau"].append(
            weights_roughness(
                weights_gaussian_independent(distances, counts, taper, 4.0), step
            )
        )
        roughness["taper"].append(
            weights_roughness(
                weights_gaussian_density(distances, counts, taper, 4), step
            )
        )
        roughness["plateau_condition"].append(
            weights_roughness(
                weights_gaussian_independent(distances, counts_wide, taper, 3.0), step
            )
        )
        roughness["taper_condition"].append(
            weights_roughness(
                weights_gaussian_density(distances, counts_wide, taper, 4), step
            )
        )
        roughness["capped"].append(
            weights_roughness(
                weights_plateau_gaussian(distances, counts, 3.0, 8.0, taper_cap=1.0),
                step,
            )
        )

    reference = max(roughness["distance"])
    assert max(roughness["plateau"]) <= 1.25 * reference
    assert max(roughness["taper"]) <= 1.5 * reference
    assert max(roughness["plateau_condition"]) <= 1.25 * reference
    assert max(roughness["taper_condition"]) <= 1.5 * reference
    assert max(roughness["capped"]) > 2.0 * reference


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "weights_model",
    [
        DistanceWeights(),
        DistanceWeights(distance_taper="nearest_neighbor"),
        DistanceWeights(distance_taper=5 * KM, waterlevel=0.1),
        DensityPlateauWeights(),
        DensityPlateauWeights(distance_taper="nearest_neighbor", waterlevel=0.1),
        DensityTaperWeights(),
        DensityTaperWeights(station_density=0.5, waterlevel=0.1),
        DensityPlateauWeights(plateau_stations=3.0, density_radius="condition_number"),
        DensityTaperWeights(density_radius="mean_interstation"),
        DensityTaperWeights(density_radius=10 * KM),
        StationDensityWeights(),
        StationDensityWeights(waterlevel=0.1),
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
    assert weights.dtype == np.float32
    assert np.all((weights > 0.0) & (weights <= 1.0))
    assert weights.min() >= weights_model.waterlevel - 1e-6

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
    # The configuration keeps the choice, the taper is calculated per network
    assert weights_model.distance_taper == distance_taper
    assert weights_model._distance_taper == pytest.approx(expected_taper)

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
        independent_station_counts(interstation_distances(subset)),
        plateau_stations=weights_model.plateau_stations,
        taper_stations=weights_model.taper_stations,
    )
    np.testing.assert_allclose(weights, expected, rtol=1e-5)

    # The cluster stations count less than the sparse stations
    independence = weights_model.get_station_counts(np.arange(len(stations)))
    assert independence[12:].max() < independence[:12].min()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "weights_model",
    [
        DistanceWeights(),
        StationDensityWeights(),
        DensityPlateauWeights(),
        DensityTaperWeights(),
    ],
)
@pytest.mark.parametrize(
    "positions_km",
    [[(0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]],
    ids=["single", "colocated"],
)
async def test_station_weights_single_site(weights_model, positions_km, octree: Octree):
    # One station, or all sensors at one site
    stations = local_stations(positions_km)
    weights_model = weights_model.model_copy(deep=True)
    weights_model.prepare(stations, octree)
    weights = await weights_model.get_weights(octree.nodes[:20], list(stations))
    assert np.all(np.isfinite(weights))
    # The same weight for all stations of a node
    np.testing.assert_allclose(weights, np.broadcast_to(weights[:, :1], weights.shape))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "weights_model",
    [
        DistanceWeights(distance_taper="nearest_neighbor"),
        StationDensityWeights(),
        DensityPlateauWeights(),
        DensityTaperWeights(),
    ],
)
async def test_station_weights_prepare_again_other_stations(
    weights_model, octree: Octree, clustered_stations: StationInventory
):
    # A second prepare with other stations resets the distances and the counts
    fresh = weights_model.model_copy(deep=True)
    weights_model = weights_model.model_copy(deep=True)
    weights_model.prepare(clustered_stations, octree)
    await weights_model.get_weights(octree.nodes[:5], list(clustered_stations))

    subset = StationInventory(stations=list(clustered_stations)[:12])
    weights_model.prepare(subset, octree)
    weights = await weights_model.get_weights(octree.nodes[:5], list(subset))

    fresh.prepare(subset, octree)
    np.testing.assert_allclose(
        weights, await fresh.get_weights(octree.nodes[:5], list(subset))
    )


@pytest.mark.asyncio
async def test_distance_weights_lazy_and_explicit(
    octree: Octree, clustered_stations: StationInventory
):
    # The default taper needs no interstation distances
    weights_model = DistanceWeights()
    weights_model.prepare(clustered_stations, octree)
    await weights_model.get_weights(octree.nodes[:5], list(clustered_stations))
    assert weights_model._interstation_distances is None

    # An explicit taper is kept as given
    weights_model = DistanceWeights(distance_taper=5 * KM)
    weights_model.prepare(clustered_stations, octree)
    assert weights_model._distance_taper == 5 * KM
    assert weights_model.distance_taper == 5 * KM


@pytest.mark.asyncio
async def test_distance_weights_prepare_again(
    octree: Octree, clustered_stations: StationInventory
):
    # The inversion prepares the weights again with other stations
    weights_model = DistanceWeights()
    weights_model.prepare(clustered_stations, octree)
    taper_all = weights_model._distance_taper
    subset = StationInventory(stations=list(clustered_stations)[12:])
    weights_model.prepare(subset, octree)
    assert weights_model._distance_taper == pytest.approx(
        2 * subset.mean_interstation_distance()
    )
    assert weights_model._distance_taper < taper_all


def test_station_weights_config():
    from pydantic import ValidationError

    from qseek.search import Search

    assert isinstance(Search().station_weights, DistanceWeights)
    assert Search.model_validate({"station_weights": None}).station_weights is None

    for model in (
        DistanceWeights(required_closest_stations=6),
        StationDensityWeights(plateau_stations=3.0, taper_cap=None),
        DensityPlateauWeights(plateau_stations=5.0, distance_taper=8 * KM),
        DensityTaperWeights(station_density=0.5, required_closest_stations=6),
        DensityPlateauWeights(density_radius="condition_number"),
    ):
        config = {"station_weights": model.model_dump(mode="json")}
        assert Search.model_validate(config).station_weights == model

    with pytest.raises(ValidationError):
        Search.model_validate({"station_weights": {"weights": "UnknownWeights"}})
    # The key of earlier versions
    with pytest.raises(ValidationError):
        Search.model_validate({"distance_weights": {"required_closest_stations": 4}})


if __name__ == "__main__":
    test_weights_gaussian_min_stations()

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qseek.distance_weights import weights_gaussian

KM = 1e3


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


if __name__ == "__main__":
    test_weights_gaussian_min_stations()

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qseek.distance_weights import weights_gaussian

KM = 1e3


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

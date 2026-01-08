import matplotlib.pyplot as plt
import numpy as np
import pytest

from qseek.distance_weights import weights_gaussian

KM = 1e3


@pytest.mark.plot
def test_weights_gaussian_min_stations():
    n_nodes = 10
    n_stations = 200
    distances = np.random.uniform(0, 100 * KM, size=(n_nodes, n_stations))

    weights_gauss = weights_gaussian(
        distances,
        distance_taper=10 * KM,
        required_stations=1,
        waterlevel=0.1,
    )

    node = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(distances[node], weights_gauss[node], label="Gaussian", alpha=0.7)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Weight")
    ax.set_title("Gaussian Weights with Minimum Stations")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    test_weights_gaussian_min_stations()

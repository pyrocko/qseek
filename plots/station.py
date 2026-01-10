import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

KM = 1e3
distances = np.linspace(0.5 * KM, 100 * KM, num=500, endpoint=True)

# distances = np.sort(rng.uniform(10 * KM, 50 * KM, 500))


def get_weights(distances, exp):
    weights = 1 / (distances**exp)
    return weights / weights.max()


def get_exp_weights(distances, radius, exp=1.5):
    distances = distances
    # radius = radius
    weights = np.exp(-(distances**exp) / (radius**exp))
    return weights / weights.max()


threshold = 1e-3

fig = plt.figure()
ax = fig.add_subplot(211)

# for exp in (0, 1, 2):
#     ax.plot(distances, get_weights(distances, exp), label=f"exp={exp}")


radius = 5 * KM
# distances = np.array([10, 12, 18, 35]) * KM
for exp in (0, 0.5, 1, 1.5, 2):
    ax.plot(
        distances,
        # get_exp_weights(distances, radius, exp=exp),
        get_weights(distances, exp=exp),
        label=f"r^exp={exp}",
    )

for exp in (0, 0.5, 1, 1.5, 2):
    ax.plot(
        distances,
        get_exp_weights(distances, radius, exp=exp),
        # get_weights(distances, exp=exp),
        ls="--",
        label=f"exp={exp}",
    )

ax.axhline(threshold, color="k", linestyle="--", label="Threshold")

ax.legend()
ax.grid(alpha=0.3)
ax.set_xlabel("Distance [km]")
ax.set_ylabel("Weight")

ax = fig.add_subplot(212)
exp = 0.5

radii = np.linspace(0.5 * KM, 20 * KM, 20)
for radius in np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0]) * KM:
    ax.plot(
        distances,
        get_exp_weights(distances, radius, exp=exp),
        label=f"radius={radius / KM:.0f} km",
    )

ax.axhline(threshold, color="k", linestyle="--", label="Threshold")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlabel("Distance [km]")
ax.set_ylabel("Weight")


plt.show()
# plt.clf()

nstations = 10
nnodes = 100
distances = rng.uniform(10 * KM, 50 * KM, size=(nnodes, nstations))
radius = distances.min(axis=1)[:, np.newaxis]

weights = np.exp(-(distances**exp) / (radius**exp))

weights /= weights.sum(axis=1)[:, np.newaxis]
print(weights.sum(axis=1))
print(weights.shape)
print(weights.dtype)

---
icon: lucide/target
---

[](){ #qseek.station_weights.StationWeightsType }

# Station weights

Station weights decide how much each station contributes to the stack of a node. Qseek calculates a weight for every pair of station and node in the search volume from their distance, see [stacking and migration](../concepts/how-it-works.md#stacking-and-migration). For every node and phase, the weights are normalized to a sum of one. Close stations get full weight: they record small events with the highest phase confidence and constrain the location best. Distant stations are tapered.

| Station weights | Full weight | Taper |
| --- | --- | --- |
| [`DistanceWeights`](#distance-weights) (default) | The `required_closest_stations` closest stations | Gaussian, absolute width: twice the mean interstation distance |
| [`StationDensityWeights`](#station-density-weights) | The closest stations up to a cumulative station weight, `plateau_weight` | Gaussian, width from the cumulative station weight `taper_weight` |
| [`LogLogisticWeights`](#log-logistic-weights) | The closest stations up to a cumulative station weight, `plateau_weight` | Log-logistic, relative to the plateau distance of each node |

`station_weights` takes one of them. With `null`, all stations get the same weight.

![Station weights](../images/station-weights.webp)
/// caption
Station weights of a node, with the defaults, for two examples of the [playground](../getting-started/playground.md). Left: Campi Flegrei, 18 stations within 12 km, node at 2 km depth below Solfatara. Right: Mount Spurr, Alaska, node at 4 km depth below sea level beneath the volcano; ten local stations within 32 km and ten regional stations from 79 km to 131 km. Bottom: the station density weights. The clustered stations count less, so the plateau of the density-based weights extends to 4.4 km at Campi Flegrei and to the most distant local station at Mount Spurr, where the ten local stations count as 4.6 independent stations. `LogLogisticWeights` tapers less than `DistanceWeights` in the dense network, and more towards the distant regional stations.
///

## Choose the station weights

Real networks are not homogeneous: dense clusters on a volcano, sparse regional stations, two sensors at one site. The station weights handle such networks differently.

- **`DistanceWeights`** count stations: the 4 closest stations of a node get full weight. In a dense cluster these are 4 stations a few hundred meters apart. The taper width is absolute and comes from the whole network: with regional stations, twice the mean interstation distance is long, and distant stations keep most of their weight.
- **`StationDensityWeights`** count independent stations. Every station gets a density weight between 0 and 1 from a Gaussian kernel density of the stations: isolated stations count 1, stations in a cluster less. The width of the kernel is the median distance between neighboring sites; co-located sensors are one site. The plateau of full weight extends to the closest stations that sum up to `plateau_weight`, 4 by default. The density weights only set the plateau and taper distances; within the plateau every station gets full weight.
- **`LogLogisticWeights`** count independent stations like `StationDensityWeights`, and taper with the distance in units of the plateau distance. The weight is 0.5 at `taper_scale` times the plateau distance, 2.2 by default. The taper follows the phase confidence of small events, which falls off with the distance to the 4th closest station in the same way on different networks. Because it is relative, the taper adapts to the station spacing around each node, and it needs no tuning between local and regional networks.

!!! tip
    Start with the default `DistanceWeights`. Try `LogLogisticWeights` when your network mixes a dense local network with distant stations, e.g. a volcano observatory with regional stations: it lowers the weight of distant stations that rarely record the small events.

### Results on the playground

The examples of the [playground](../getting-started/playground.md), each searched with the three station weights and their defaults, `MADTrigger` with `mad_factor` 10:

| Example | Station weights | Detections | With ≥ 8 picks | Residual RMS, median | Catalog events matched | Epicenter offset, median |
| --- | --- | --- | --- | --- | --- | --- |
| Campi Flegrei, 1 day, 18 stations | `DistanceWeights` | 732 | 521 | 0.284 s | 45 / 45 | 241 m |
| | `StationDensityWeights` | 728 | 510 | 0.283 s | 45 / 45 | 256 m |
| | `LogLogisticWeights` | 772 | 521 | 0.289 s | 45 / 45 | 248 m |
| Campi Flegrei, 10 days, 19 stations | `DistanceWeights` | 6379 | 3750 | 0.278 s | 209 / 211 | 280 m |
| | `StationDensityWeights` | 6228 | 3692 | 0.279 s | 209 / 211 | 269 m |
| | `LogLogisticWeights` | 6724 | 3758 | 0.280 s | 210 / 211 | 310 m |
| Mount Spurr, 3 days, 10 local and 10 regional stations | `DistanceWeights` | 1758 | 1046 | 0.349 s | 228 / 234 | 416 m |
| | `StationDensityWeights` | 1744 | 1043 | 0.354 s | 228 / 234 | 441 m |
| | `LogLogisticWeights` | 2063 | 1121 | 0.375 s | 229 / 234 | 436 m |

`LogLogisticWeights` detects the most events in all three examples, 5% to 17% more than `DistanceWeights`, and the most with at least 8 picks: 75 more at Mount Spurr, where `DistanceWeights` keep the regional stations at 80 km to 85 km at a weight of about 0.6, `LogLogisticWeights` at about 0.35. The catalog offsets of the additional small events are slightly larger, and so is the median residual. `StationDensityWeights` with its defaults detects slightly fewer events than `DistanceWeights`: its taper is wider. At Campi Flegrei, the 18 stations count as 11.1 independent stations, less than `taper_weight`, so the most distant station sets the taper width.

## Distance weights

The `required_closest_stations` closest stations of a node get full weight, 4 by default. More distant stations are tapered with a Gaussian function; `distance_taper` is its full width at half maximum. By default it adapts to the network: twice the mean interstation distance. `waterlevel` keeps a minimum weight for distant stations; with the default `0.0`, stations far outside the taper do not contribute.

```python exec='on'
from qseek.utils import json_example
from qseek.station_weights import DistanceWeights

print(json_example(DistanceWeights()))
```

<div class="qs-config" markdown>

::: qseek.station_weights.DistanceWeights
    options:
      heading_level: 3

</div>

## Station density weights

The closest stations of a node get full weight until their density weights sum up to `plateau_weight`, 4 by default. Beyond this plateau distance, a Gaussian taper decays; its standard deviation is half the distance at which the cumulative density weight reaches `taper_weight`, 12 by default. When the network has less density weight in total than `taper_weight`, the most distant station sets the width; Qseek logs the total weight of the network when the search starts.

The density weights depend on the available stations. When a station has no data in a window, Qseek calculates the density weights of the remaining stations.

```python exec='on'
from qseek.utils import json_example
from qseek.station_weights import StationDensityWeights

print(json_example(StationDensityWeights()))
```

<div class="qs-config" markdown>

::: qseek.station_weights.StationDensityWeights
    options:
      heading_level: 3

</div>

## Log-logistic weights

The plateau distance $d_p$ of a node is the distance at which the density weights of the closest stations sum up to `plateau_weight`, as for the station density weights. A station at distance $d$ gets the weight

$$
w(d) = \frac{1}{1 + \left(\dfrac{d}{s \, d_p}\right)^{n}}
$$

with the `taper_scale` $s$, 2.2 by default, and the `taper_exponent` $n$, 4 by default. Stations within the plateau get almost full weight, 0.96 at the plateau distance. The weight is 0.5 at 2.2 times the plateau distance and 0.13 at 3.5 times.

The defaults come from the phase confidences of three playground runs: on Campi Flegrei (1 day and 10 days) and Mount Spurr, the mean PhaseNet confidence of the stations falls to half at 2.1 to 2.4 times the distance of the 4th closest station, with exponents of 3.2 to 4.4. Raise `taper_scale` to let more distant stations contribute, e.g. when your events are larger than the smallest events of the catalog.

```python exec='on'
from qseek.utils import json_example
from qseek.station_weights import LogLogisticWeights

print(json_example(LogLogisticWeights()))
```

<div class="qs-config" markdown>

::: qseek.station_weights.LogLogisticWeights
    options:
      heading_level: 3

</div>

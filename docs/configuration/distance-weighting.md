---
icon: lucide/target
---

# Distance weighting

Distance weights decide how much each station contributes to the stack of a node. The closest stations of a node constrain its location best, so they get full weight; more distant stations are tapered with a Gaussian decay. Qseek calculates a weight for every pair of station and node in the search volume, see [stacking and migration](../concepts/how-it-works.md#stacking-and-migration).

![Distance weighting](../images/distance-weights.webp)
/// caption
Weights of the stations for a single node (dots) and their cumulative weight (line). Three parameters shape the weights: (1) the number of closest stations with full weight, (2) the taper distance and (3) the waterlevel.
///

- `required_closest_stations` get full weight, 4 by default.
- `distance_taper` sets how fast the weight of more distant stations decays. By default it adapts to the network: twice the mean interstation distance.
- `waterlevel` keeps a minimum weight for distant stations. With the default `0.0`, stations far outside the taper do not contribute.

!!! tip
    The defaults suit local and regional networks. If distant stations should still contribute to the stack, raise the `waterlevel`.

```python exec='on'
from qseek.utils import json_example
from qseek.distance_weights import DistanceWeights

print(json_example(DistanceWeights()))
```

<div class="qs-config" markdown>

::: qseek.distance_weights.DistanceWeights
    options:
      heading_level: 3

</div>

# Station Distance Weighting

Station distance weights are used to give stations close to a node higher weight as it contributes more accurate information about the event's hypocenter. Station nodes are calculated for every node and station in the search volume.

!!! tip
    Station weights play a crucial role for successful detection and location of events.
    While the default parameters are tuned for local and regional networks. The `distance_taper` and `waterlevel` may have to be tuned for different layouts.

![Distance Weighting](../images/distance-weights.webp)
/// caption
Distance weights beteen a single node and stations. Shown here is the station weight and cummulative weight for the whole network. With three tuneable parameters: (1) Required closest stations, (2) taper distance and (3) waterlevel.
///

```python exec='on'
from qseek.utils import generate_docs
from qseek.distance_weights import DistanceWeights

print(generate_docs(DistanceWeights()))
```

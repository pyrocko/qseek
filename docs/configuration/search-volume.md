---
icon: lucide/boxes
---

# Search volume

The search volume is the region where Qseek looks for earthquakes; you configure it in the `octree` of the search. You place it with a center `location` and its bounds in east, north and depth direction, relative to the center. Qseek divides the volume into root nodes and refines the nodes around detected events, see [octree refinement](../concepts/how-it-works.md#octree-refinement).

![Octree refinement](../images/octree-concept.webp)
/// caption
The octree refines around a seismic source, from level 0 with 5977 nodes to level 2 with 6812 nodes. Map view (top) and depth section (bottom) of the semblance; the cross marks the maximum.
///

## Set up the volume

- **Center:** set `location` to the center of your network or of the expected seismicity. The location must not be `0, 0`.
- **Bounds:** `east_bounds`, `north_bounds` and `depth_bounds` are in meters, relative to the center. Depth is positive down. Leave a margin around the seismicity: detections at the border of the volume are ignored, see [`ignore_boundary`][qseek.search.Search.ignore_boundary].
- **Root nodes:** every extent must be a multiple of `root_node_size`. The number of root nodes sets the cost of the coarse search: the default 20 km × 20 km × 20 km volume with 1 km root nodes has 8000 root nodes.
- **Resolution:** the smallest node size is `root_node_size / 2**(n_levels - 1)`. The defaults, 1 km root nodes and 5 levels, refine down to 62.5 m.

!!! tip
    Qseek stacks every root node in every window, but refines only the nodes around detections. For a finer resolution, add levels rather than shrinking the root nodes.

```python exec='on'
from qseek.utils import json_example
from qseek.models.location import Location
from qseek.octree import Octree

print(json_example(Octree(location=Location(lat=52.38, lon=13.06))))
```

<div class="qs-config" markdown>

::: qseek.octree.Octree
    options:
      heading_level: 3

</div>

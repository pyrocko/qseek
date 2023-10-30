# Octree

A 3D space is searched for sources of seismic energy. Lassie created an octree structure which is iteratively refined when energy is detected, to focus on the source' location. This speeds up the search and improves the resolution of the localisations.

```python exec='on'
from lassie.utils import generate_docs
from lassie.octree import Octree

print(generate_docs(Octree()))
```

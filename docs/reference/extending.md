---
icon: lucide/puzzle
---

# Extending Qseek

You can add your own modules to Qseek, e.g. a ray tracer for your velocity model, an image function or a data provider. There are two ways:

- **Callback scripts:** a single Python file with a [custom callback](../configuration/callbacks.md#custom-callbacks), e.g. to send alerts or write detections to a database. No package needed.
- **Module plugins:** a Python package that provides new modules of any kind. Once installed, the modules are available in the configuration like the modules of Qseek.

## Module plugins

A module plugin subclasses one of the [module base classes](api/modules.md) and registers its package as an entry point in the `qseek.modules` group. When Qseek is imported, it imports all installed plugins, so their modules can be selected in the configuration by their discriminator field, e.g. `"tracer": "StraightRayTracer"`. A plugin that fails to import is skipped with a warning.

This example plugin adds a ray tracer for straight rays with a P and an S velocity:

```text
qseek-example/
├── pyproject.toml
└── qseek_example/
    ├── __init__.py
    └── tracer.py
```

```toml title="pyproject.toml"
[project]
name = "qseek-example"
version = "0.1.0"
dependencies = ["qseek"]

[project.entry-points."qseek.modules"]
example = "qseek_example"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"
```

```python title="qseek_example/__init__.py"
from qseek_example.tracer import StraightRayTracer

__all__ = ["StraightRayTracer"]
```

```python title="qseek_example/tracer.py"
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from pydantic import Field, PositiveFloat

from qseek.tracers.base import ModelledArrival, RayTracer

if TYPE_CHECKING:
    from qseek.models.location import Location
    from qseek.models.station import Station
    from qseek.octree import Node


class StraightRayTracer(RayTracer):
    """Travel times along straight rays for a P and an S velocity."""

    tracer: Literal["StraightRayTracer"] = "StraightRayTracer"
    velocity_p: PositiveFloat = Field(default=6000.0, description="P velocity in m/s.")
    velocity_s: PositiveFloat = Field(default=3500.0, description="S velocity in m/s.")

    def get_available_phases(self) -> tuple[str, ...]:
        return ("straight:P", "straight:S")

    def _velocity(self, phase: str) -> float:
        return {"straight:P": self.velocity_p, "straight:S": self.velocity_s}[phase]

    def get_travel_time_location(
        self, phase: str, source: Location, receiver: Location
    ) -> float:
        return source.distance_to(receiver) / self._velocity(phase)

    async def get_travel_times(
        self, phase: str, nodes: Sequence[Node], stations: Sequence[Station]
    ) -> np.ndarray:
        # Travel times in seconds, shape (n_nodes, n_stations)
        return np.array(
            [
                [node.as_location().distance_to(station) for station in stations]
                for node in nodes
            ]
        ) / self._velocity(phase)

    def get_arrivals(
        self,
        phase: str,
        event_time: datetime,
        source: Location,
        receivers: Sequence[Location],
    ) -> list[ModelledArrival | None]:
        travel_times = self.get_travel_times_locations(phase, source, receivers)
        return [
            ModelledArrival(phase=phase, time=event_time + timedelta(seconds=t))
            for t in travel_times
        ]
```

The discriminator field, here `tracer`, needs a `Literal` with a name that is unique among all ray tracers. The class docstring and the field descriptions are shown by `qseek modules` and in the configuration.

## Install and use the plugin

Install the plugin into the environment of Qseek and check that the module is available. `qseek modules` marks plugin modules with the name of the plugin:

```sh
pip install -e qseek-example/
qseek modules
```

Then use the module in the configuration:

```json title="The plugin's ray tracer"
"ray_tracers": [
  {"tracer": "StraightRayTracer", "velocity_p": 6000.0, "velocity_s": 3500.0}
]
```

The phase descriptions of the ray tracer, here `straight:P` and `straight:S`, go into the `phase_map` of the [image function](../configuration/image-functions.md).

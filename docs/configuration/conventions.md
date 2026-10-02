---
icon: lucide/ruler
---

# Conventions

These formats apply to all fields of the configuration.

[](){ #qseek.types.FilePath }
[](){ #qseek.types.DirectoryPath }

## Paths

Paths are absolute or relative to the directory where you start `qseek`. Start the search in the directory of the configuration file, so that relative paths in the configuration point to the right files. Qseek checks that every given file and directory exists when it reads the configuration.

## Units

Distances, depths and elevations are in meters, times in seconds, frequencies in Hz. Depth is positive down.

[](){ #qseek.utils.DateTime }

## Date and time

Dates and times follow [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) and need a time zone, e.g. `Z` or `+00:00` for UTC. A date without a time is midnight UTC. `"now"`, `"today"` and `"yesterday"` are also valid.

```json title="Dates and times"
{
  "start_time": "2023-10-28T01:21:21.003Z",
  "end_time": "2023-10-29"
}
```

## Durations

Durations follow ISO 8601 as well: `"PT5M"` is 5 minutes, `"PT600S"` 600 seconds, `"PT1H30M"` one and a half hours.

[](){ #qseek.utils.NSLType }

## Station codes

Stations are identified by their network, station and location code (NSL), written as `"<network>.<station>.<location>"`, e.g. `"6A.STA13.00"`. The location code is often empty: `"GE.RUE."`. Network and location codes have up to two characters, station codes up to five.

Where a field selects stations, such as the `exclude_stations` of the [stations](stations.md), partial codes and wildcards match several stations: `"6A."` matches the whole network, `"6A.STA*"` all stations starting with `STA`.

[](){ #qseek.utils.PhaseDescription }

## Phase descriptions

A phase description names a phase of a ray tracer as `"<tracer>:<phase>"`, e.g. `"cake:P"`, `"fm:S"` or `"constant:P"`. The `phase_map` of the [image function](image-functions.md) assigns its P and S images to these phase descriptions, and every phase needs a [ray tracer](ray-tracers.md) that calculates its travel times.

## Locations

A location is a geographic reference, `lat` and `lon` in degrees, plus a shift to the east and north in meters. The search volume and the velocity models are placed with locations.

```python exec='on'
from qseek.utils import json_example
from qseek.models.location import Location

print(json_example(Location(lat=52.3825, lon=13.0644)))
```

<div class="qs-config" markdown>

::: qseek.models.location.Location
    options:
      heading_level: 3

</div>

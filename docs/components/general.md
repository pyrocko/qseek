# General Settings

## Paths

Paths can be relative to the location of the config file or absolute. **File paths and directory paths are checked whether they exist**.

## Date and Time

Serialisation of time, dates and date times and durations follow [ISO8601](https://en.wikipedia.org/wiki/ISO_8601) format with timezone information. E.g. `2023-10-28T01:21:21.003Z`.

Duration are serialized like `PT600S`, this example shows a duration of 600 seconds, 10 minutes.


!!!+ note Timezone Info
    All datetimes are timezone aware! For UTC this is the `Z` or `+00:00` suffix.
    Also other offsets can be defined according to [ISO8601](https://en.wikipedia.org/wiki/ISO_8601).

```json title="Example of datetimes and durations"
{
    "start_time": "2023-10-28T01:21:21.003+00:00",
    "end_time": "2023-10-28T01:21:21.003+00:00",
    "duration": "PT600S"
}
```

## Locations

Geographic locations have a geographic reference and a relative shift in meters. The octree or velocity models are referenced using Location objects.

All **distances, depths and elevations are given in meters**.

```python exec='on'
from qseek.utils import generate_docs
from qseek.models.location import Location

print(generate_docs(Location(lat= 52.3825, lon=13.0644)))
```

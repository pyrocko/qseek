---
icon: lucide/radio-tower
---

# Stations

Qseek needs the location and the code of every station. Load them from [StationXML](https://www.fdsn.org/xml/station/) or [Pyrocko station YAML](https://pyrocko.org/docs/current/formats/yaml.html) files:

```json title="Stations from StationXML"
"stations": {
  "station_xmls": ["meta/stations.xml", "meta/temporary-network/"],
  "exclude_stations": ["GE.RUE.", "6A.STA*"]
}
```

`station_xmls` takes files and directories; Qseek loads all `.xml` files of a directory. Pyrocko station YAML files go into `pyrocko_station_yamls`. When the search starts, Qseek removes stations without waveform data.

- **Exclude stations** with `exclude_stations`, as [station codes](conventions.md#station-codes). Partial codes and wildcards exclude whole networks or groups of stations.
- **Limit the network** with `max_distance` in meters from the center of the stations, e.g. to ignore distant stations of a regional network.

!!! note "Instrument responses"
    [Magnitudes](magnitudes.md) and [ground motions](features.md) are measured on restituted waveforms. They need the instrument responses from StationXML.

```python exec='on'
from qseek.utils import json_example
from qseek.models.station import StationInventory

print(json_example(StationInventory()))
```

<div class="qs-config" markdown>

::: qseek.models.station.StationInventory
    options:
      heading_level: 3

</div>

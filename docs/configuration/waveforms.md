---
icon: lucide/audio-waveform
---

[](){ #qseek.waveforms.providers.WaveformProviderType }

# Waveforms

The `data_provider` delivers the waveforms to the search. Qseek loads them in windows while the search runs, so data sets of any size work. Choose the provider by how your data is stored:

| Provider | Use for |
| --- | --- |
| [`SDSArchive`](#sds-archive) | MiniSEED files in an SDS archive. The fastest provider and the default. |
| [`PyrockoSquirrel`](#pyrocko-squirrel) | Waveform files in any structure and any format that Pyrocko reads. |
| [`SeedLink`](#seedlink) | Real-time streams from SeedLink servers. |

!!! tip "Download data from FDSN data centers"
    [FDSN Rush](https://miili.github.io/FDSN-rush/) downloads data from [FDSN](https://www.fdsn.org/) data centers into an SDS archive, together with the StationXML metadata. The [quick start](../getting-started/quick-start.md) uses it.

## SDS archive

Reads MiniSEED files from a [SeisComP Data Structure (SDS)](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html) archive, organized by year, network, station and channel. Set the `archive` directory and limit the search to a time span with `start_time` and `end_time`.

```python exec='on'
from qseek.utils import json_example
from qseek.waveforms.sds import SDSArchive

print(json_example(SDSArchive.model_construct()))
```

<div class="qs-config" markdown>

::: qseek.waveforms.sds.SDSArchive
    options:
      heading_level: 3

</div>

## Pyrocko Squirrel

Reads waveform files in any directory structure and any format supported by Pyrocko, through the [Squirrel](https://pyrocko.org/docs/current/topics/squirrel.html) data access framework. List the directories in `waveform_dirs`. For large data sets, a `persistent` collection keeps the file index between runs and speeds up the start.

```python exec='on'
from qseek.utils import json_example
from qseek.waveforms.squirrel import PyrockoSquirrel

print(json_example(PyrockoSquirrel(persistent="docs")))
```

<div class="qs-config" markdown>

::: qseek.waveforms.squirrel.PyrockoSquirrel
    options:
      heading_level: 3

</div>

## SeedLink

Streams waveforms from SeedLink servers for real-time detection and localization. The [real-time monitoring guide](../guides/real-time.md) shows a complete setup with alerts for new detections. With `sds_archive`, Qseek also writes the received data into an SDS archive.

!!! note
    The SeedLink provider needs `slinktool` from [libslink](https://github.com/EarthScope/libslink).

```python exec='on'
from qseek.utils import json_example
from qseek.waveforms.seedlink.seedlink import SeedLink

print(json_example(SeedLink()))
```

<div class="qs-config" markdown>

::: qseek.waveforms.seedlink.seedlink.SeedLink
    options:
      heading_level: 3

</div>

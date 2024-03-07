# Seismic Data

## Waveform Data

The seismic can be delivered in MiniSeed or any other format compatible with Pyrocko. Lassie utilizes the Pyrocko Squirrel for fast and asynchronous data access.

To prepare your data for EQ detection and localisation, **organize it in a MiniSeed file or an [SDS structure](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html)**.

```python exec='on'
from lassie.utils import generate_docs
from lassie.waveforms import PyrockoSquirrel

print(generate_docs(PyrockoSquirrel()))
```

## Meta Data

Meta data is required primarily for **station locations and codes**.

Supported data formats are:

* [x] [StationXML](https://www.fdsn.org/xml/station/)
* [x] [Pyrocko Station YAML](https://pyrocko.org/docs/current/formats/yaml.html)

Metadata does not need to include response information for pure detection and localisation. If local magnitudes M~L~ are extracted, response information is required.

```python exec='on'
from lassie.utils import generate_docs
from lassie.models.station import Stations

print(generate_docs(Stations()))
```

# Waveform Data

The seismic can be delivered in MiniSeed or any other format compatible with Pyrocko. Qseek utilizes the Pyrocko Squirrel for fast and asynchronous data access.

To prepare your data for EQ detection and localisation, **organize it in a MiniSeed file or an [SDS structure](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html)**.

## Local Waveform Data

To copy large data from [FDSN](https://www.fdsn.org/) data centers use [fdsn-fetch](https://github.com/miili/fdsn-fetch).

```python exec='on'
from qseek.utils import generate_docs
from qseek.waveforms.squirrel import PyrockoSquirrel

print(generate_docs(PyrockoSquirrel(persistent="docs")))
```

## Real-Time Waveform Streaming

```python exec='on'
from qseek.utils import generate_docs
from qseek.waveforms.seedlink import SeedLink

print(generate_docs(SeedLink(persistent="docs")))
```

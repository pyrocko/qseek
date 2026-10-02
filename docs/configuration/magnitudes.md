---
icon: lucide/gauge
---

[](){ #qseek.magnitudes.EventMagnitudeCalculatorType }

# Magnitudes

Qseek calculates magnitudes for every detection. `magnitudes` is a list, so you can calculate several magnitudes at once, e.g. a local magnitude and a moment magnitude.

```json title="Local and moment magnitude"
"magnitudes": [
  {"magnitude": "LocalMagnitude", "model": "iceland-reykjanes"},
  {"magnitude": "MomentMagnitude", "gf_store_dirs": ["gf-stores/"]}
]
```

Both measure peak amplitudes on restituted waveforms, so the [stations](stations.md) need instrument responses from StationXML. A station magnitude needs a signal-to-noise ratio above `min_signal_noise_ratio`; the network magnitude needs at least `min_stations` station magnitudes.

## Local magnitude

The local magnitude (ML) uses a regional attenuation model. Choose the `model` of your region:

```python exec='on'
from typing import get_args

from qseek.magnitudes.local_magnitude_models import LocalMagnitudeModel, ModelName

names = set(get_args(ModelName))
rows = [
    "| Model | Reference | Distance range | Amplitude |",
    "| --- | --- | --- | --- |",
]
for model in LocalMagnitudeModel.__subclasses__():
    name = model.model_name()
    if name not in names:
        continue
    url = f"https://doi.org/{model.doi}" if model.doi else model.reference
    reference = f"[{model.author}]({url})" if url else model.author
    if model.epicentral_range:
        rng, kind = model.epicentral_range, "epicentral"
    else:
        rng, kind = model.hypocentral_range, "hypocentral"
    distance = f"{rng.min / 1e3:g}–{rng.max / 1e3:g} km {kind}" if rng else ""
    rows.append(f"| `{name}` | {reference} | {distance} | {model.max_amplitude} |")
print("\n".join(rows))
```

If no model fits, define your own attenuation with a `CustomLocalMagnitudeModel` as `model`. For each station, matched by its [station code](conventions.md#station-codes) with wildcards, an attenuation model gives

$$
M_L = \log_{10} A + a \log_{10} r + b\,r + c
$$

with the amplitude $A$ (Wood-Anderson amplitude in mm) and the epicentral or hypocentral distance $r$ in km. This example writes the relation of Hutton and Boore (1987) for all stations:

```json title="Custom local magnitude model"
"magnitudes": [
  {
    "magnitude": "LocalMagnitude",
    "model": {
      "distance": "hypocentral",
      "max_amplitude": "wood-anderson",
      "attenuation_models": {
        "*": {"a": 1.11, "b": 0.00189, "c": 0.591}
      }
    }
  }
]
```

```python exec='on'
from qseek.utils import json_example
from qseek.magnitudes.local_magnitude import LocalMagnitude

print(json_example(LocalMagnitude()))
```

<div class="qs-config" markdown>

::: qseek.magnitudes.local_magnitude.LocalMagnitude
    options:
      heading_level: 3

</div>

## Moment magnitude

The moment magnitude (Mw) compares the observed peak amplitudes with peak amplitudes modeled from [Pyrocko GF stores](https://pyrocko.org/docs/current/topics/pyrocko-gf.html), Green's function databases for your velocity model. The method is described in [Dahm et al., 2024](https://doi.org/10.26443/seismica.v3i2.1205).

```python exec='on'
from qseek.utils import json_example
from qseek.magnitudes.moment_magnitude import MomentMagnitude

print(json_example(MomentMagnitude()))
```

<div class="qs-config" markdown>

::: qseek.magnitudes.moment_magnitude.MomentMagnitude
    options:
      heading_level: 3

</div>

## Recalculate magnitudes

To calculate the magnitudes of an existing run again, e.g. after changing the `magnitudes`, see [recalculate magnitudes and features](../guides/manage-runs.md#recalculate-magnitudes-and-features).

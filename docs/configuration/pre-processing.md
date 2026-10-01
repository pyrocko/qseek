---
icon: lucide/filter
---

[](){ #qseek.pre_processing.module.PreProcessing }

# Pre-processing

Before the image function annotates the waveforms, Qseek pre-processes them. `pre_processing` is a list of steps, applied in order. By default, the waveforms are resampled to 100 Hz and bandpass filtered between 0.5 and 30 Hz:

```json title="Default pre-processing"
"pre_processing": [
  {"process": "resample", "sampling_frequency": 100.0},
  {"process": "bandpass", "bandpass": [0.5, 30.0]}
]
```

Each step applies to all stations, or only to the [station codes](conventions.md#station-codes) in its `stations` list.

## Resampling

The image function needs one sampling rate for all stations. If your stations record at different rates, resample them.

- `resample` uses polyphase filtering and reaches any target rate, higher or lower.
- `downsample` decimates by integer factors. It only lowers the sampling rate.

```python exec='on'
from qseek.utils import json_example
from qseek.pre_processing.resample import Resample

print(json_example(Resample()))
```

<div class="qs-config" markdown>

::: qseek.pre_processing.resample.Resample
    options:
      heading_level: 3

</div>

```python exec='on'
from qseek.utils import json_example
from qseek.pre_processing.resample import Downsample

print(json_example(Downsample()))
```

<div class="qs-config" markdown>

::: qseek.pre_processing.resample.Downsample
    options:
      heading_level: 3

</div>

## Frequency filters

Butterworth filters remove noise outside the frequency band of the earthquakes. With `zero_phase`, the bandpass does not shift the phase arrivals.

```python exec='on'
from qseek.utils import json_example
from qseek.pre_processing.frequency_filters import Bandpass

print(json_example(Bandpass()))
```

<div class="qs-config" markdown>

::: qseek.pre_processing.frequency_filters.Bandpass
    options:
      heading_level: 3

</div>

```python exec='on'
from qseek.utils import json_example
from qseek.pre_processing.frequency_filters import Highpass

print(json_example(Highpass()))
```

<div class="qs-config" markdown>

::: qseek.pre_processing.frequency_filters.Highpass
    options:
      heading_level: 3

</div>

```python exec='on'
from qseek.utils import json_example
from qseek.pre_processing.frequency_filters import Lowpass

print(json_example(Lowpass()))
```

<div class="qs-config" markdown>

::: qseek.pre_processing.frequency_filters.Lowpass
    options:
      heading_level: 3

</div>

## Denoising

The DeepDenoiser neural network removes noise from the waveforms. It is slow; use it for noisy stations and run it on a GPU.

```python exec='on'
from qseek.utils import json_example
from qseek.pre_processing.deep_denoiser import DeepDenoiser

print(json_example(DeepDenoiser()))
```

<div class="qs-config" markdown>

::: qseek.pre_processing.deep_denoiser.DeepDenoiser
    options:
      heading_level: 3

</div>

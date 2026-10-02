---
icon: lucide/activity
---

[](){ #qseek.images.ImageFunctionType }

# Image functions

The image function turns the waveforms of each station into *images* of the P and S phase arrivals. Qseek stacks these images along the travel times to detect and locate earthquakes, see [How Qseek works](../concepts/how-it-works.md#phase-images). After a detection, the `picker` of the image function picks the phase arrivals around their modeled arrival times.

| Image function | Image | Use for |
| --- | --- | --- |
| [`SeisBench`](#seisbench) | Phase arrival probability of a machine learning picker | Most data sets. The default. |
| [`StaLta`](#stalta) | Logarithm of the STA/LTA ratio | Data where no pre-trained model fits, or without a GPU. |

The `phase_map` assigns the P and S images to the [phase descriptions](conventions.md#phase-descriptions) of the ray tracers, e.g. `{"P": "cake:P", "S": "cake:S"}`. With `weights`, one phase can contribute more to the stack than the other.

## SeisBench

[SeisBench](https://github.com/seisbench/seisbench) provides machine learning phase pickers, pre-trained on different data sets. The image is the probability of a P or S phase arrival.

- **Model:** PhaseNet is a good start. EQTransformer, OBSTransformer (ocean bottom seismometers) and LFEDetect (low-frequency earthquakes) suit specific settings.
- **Pre-trained weights:** choose weights trained on data similar to yours, e.g. `"instance"` for Italy or `"stead"` for global data.
- **GPU:** with `torch_use_cuda`, the annotation runs on a CUDA GPU, which is much faster than on the CPU for large data sets.

```python exec='on'
from qseek.utils import json_example
from qseek.images.seisbench import SeisBench

print(json_example(SeisBench()))
```

<div class="qs-config" markdown>

::: qseek.images.seisbench.SeisBench
    options:
      heading_level: 3

</div>

### Sampling rate

The pre-trained models are trained at a fixed sampling rate, e.g. 100 Hz for PhaseNet. The [`sampling_rate`][qseek.images.seisbench.SeisBench.sampling_rate] field sets the rate that the model assumes for its input. It matters because the model sees the waveform in samples: it recognizes a phase onset by its shape over a fixed number of samples. If the assumed rate differs from the real rate of the data, the model sees the waveform stretched or compressed in time.

- **`"input"` (default):** Qseek takes the rate from the input traces. The model sees the data at its real rate. All traces of the search need the same rate, otherwise Qseek raises an error. If your stations record at different rates, [resample](pre-processing.md#resampling) them to one rate.
- **A rate in Hz, e.g. `100.0`:** The model assumes this rate for all traces. A rate above the native rate of the model, e.g. 200 Hz for a model trained at 100 Hz, rescales the input by their ratio. This can help to detect high-frequency microseismic events. Make sure that the data actually has this rate, otherwise the model sees distorted phases.

The rescaling by the ratio of assumed rate to native rate also narrows the detection window by the same ratio (0.2 s at the native rate). The blinding of the image function uses the assumed rate. With `"input"`, it uses the native rate of the model until the first traces are processed.

### SeisBench picker

```python exec='on'
from qseek.utils import json_example
from qseek.images.seisbench import AnnotationPicker

print(json_example(AnnotationPicker()))
```

<div class="qs-config" markdown>

::: qseek.images.seisbench.AnnotationPicker
    options:
      heading_level: 3

</div>

!!! abstract "Citations"
    Woollam, J., Münchmeyer, J., Tilmann, F., Rietbrock, A., Lange, D., Bornstein, T., et al. (2022). SeisBench — A toolbox for machine learning in seismology. Seismological Research Letters, 93(3), 1695–1709. [https://doi.org/10.1785/0220210324](https://doi.org/10.1785/0220210324)

    Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International, 216(1), 261–273. [https://doi.org/10.1093/gji/ggy423](https://doi.org/10.1093/gji/ggy423)

    Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer — an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications, 11, 3952. [https://doi.org/10.1038/s41467-020-17591-w](https://doi.org/10.1038/s41467-020-17591-w)

## STA/LTA

A short-term average to long-term average (STA/LTA) characteristic function, adapted from [QuakeMigrate](https://github.com/QuakeMigrate/QuakeMigrate). P phases are detected on the vertical component, S phases on the horizontal components.

The centered STA/LTA places the short-term window after the long-term window, so the ratio peaks at the phase onset. The image is the logarithm of the STA/LTA ratio: stacking the images yields the geometric mean of the ratios. The semblance and the detection threshold are in log units, noise is at 0.

```python exec='on'
from qseek.utils import json_example
from qseek.images.sta_lta import StaLta

print(json_example(StaLta()))
```

<div class="qs-config" markdown>

::: qseek.images.sta_lta.StaLta
    options:
      heading_level: 3

</div>

### STA/LTA picker

```python exec='on'
from qseek.utils import json_example
from qseek.images.sta_lta import StaLtaPicker

print(json_example(StaLtaPicker()))
```

<div class="qs-config" markdown>

::: qseek.images.sta_lta.StaLtaPicker
    options:
      heading_level: 3

</div>

!!! abstract "Citation"
    Winder, T., Bacon, C. A., Smith, J. D., Hudson, T., Greenfield, T., & White, R. S. (2020). QuakeMigrate: a modular, open-source Python package for automatic earthquake detection and location. AGU Fall Meeting 2020.

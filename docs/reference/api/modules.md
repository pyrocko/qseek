---
icon: lucide/puzzle
---

# Module base classes

Every module of a search, e.g. a ray tracer or an image function, subclasses one of these base classes. Subclass them to [extend Qseek](../extending.md) with your own modules.

| Base class | Configured in | Discriminator field |
| --- | --- | --- |
| `WaveformProvider` | `data_provider` | `provider` |
| `BatchPreProcessing` | `pre_processing` | `process` |
| `ImageFunction` | `image_function` | `image` |
| `Picker` | `picker` of the image function | |
| `RayTracer` | `ray_tracers` | `tracer` |
| `TravelTimeCorrections` | `station_corrections` | `corrections` |
| `Trigger` | `trigger` | `trigger` |
| `EventMagnitudeCalculator` | `magnitudes` | `magnitude` |
| `FeatureExtractor` | `features` | `feature` |
| `Callback` | `callbacks` | `callback` |

::: qseek.waveforms.base.WaveformProvider
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.pre_processing.base.BatchPreProcessing
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.images.base.ImageFunction
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.images.base.Picker
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.tracers.base.RayTracer
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.tracers.base.ModelledArrival
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.corrections.base.TravelTimeCorrections
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.triggers.Trigger
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.magnitudes.base.EventMagnitudeCalculator
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.features.base.FeatureExtractor
    options:
      heading_level: 2
      show_if_no_docstring: true

::: qseek.plugins.callback.Callback
    options:
      heading_level: 2
      show_if_no_docstring: true

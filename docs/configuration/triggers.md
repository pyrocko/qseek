---
icon: lucide/activity
---

[](){ #qseek.triggers.TriggerType }

# Triggers

The trigger turns the detection function, the maximum semblance over all nodes, into detections. A peak of the detection function is a detection when its height and its prominence exceed the thresholds of the trigger. Peaks closer than the `blinding`, 1 s by default, count as one detection. See [detection](../concepts/how-it-works.md#detection).

| Trigger | Threshold | Adapts to |
| --- | --- | --- |
| [`MADTrigger`](#mad-trigger) (default) | `mad_factor` times the median absolute deviation (MAD) of the window | Each window |
| [`ModZScoreTrigger`](#modified-z-score-trigger) | `z_score` scaled MADs above the median of the window | Each window, and its noise floor |
| [`ThresholdTrigger`](#fixed-threshold) | Fixed semblance `threshold` | Nothing: the same threshold for all windows |

Qseek computes the thresholds of the adaptive triggers in every processed window of [`window_length`][qseek.search.Search.window_length], 5 minutes by default.

We recommend an adaptive trigger. On one day of the 2024 Campi Flegrei swarm (18 INGV stations, 287 windows), the adaptive thresholds range from 0.03 in quiet windows to 0.25 in the busiest windows:

| Trigger | Detections | With ≥ 8 picks | Median picks | INGV events matched |
| --- | --- | --- | --- | --- |
| `ThresholdTrigger`, `threshold` 0.3 | 315 | 315 | 20 | 45 / 45 |
| `MADTrigger`, `mad_factor` 10 | 732 | 521 | 12 | 45 / 45 |
| `ModZScoreTrigger`, `z_score` 6 | 732 | 524 | 12 | 45 / 45 |

Both adaptive triggers find all detections of the fixed threshold, at the same locations, and about 200 more with at least 8 picks. The fixed threshold of 0.3 misses weak events in quiet windows, where the adaptive thresholds are six to ten times lower. On this data set, `z_score` 6 gives nearly the same detections as `mad_factor` 10: the noise floor of the detection function is low, about 0.01. The default `z_score` 7 is slightly stricter, with 642 detections.

!!! tip
    The median absolute deviation rises with the seismicity rate: during a swarm, the events themselves raise the threshold of their window. Lower `mad_factor` or `z_score` to detect more events in busy windows. On Campi Flegrei, `mad_factor` 8 to 12 does not change the number of detections in windows with a threshold below 0.05.

## MAD trigger

The threshold is `mad_factor` times the median absolute deviation (MAD) of the detection function in each window, 10 by default. It is the minimum height and the minimum prominence of a detection.

```python exec='on'
from qseek.utils import json_example
from qseek.triggers import MADTrigger

print(json_example(MADTrigger()))
```

<div class="qs-config" markdown>

::: qseek.triggers.MADTrigger
    options:
      heading_level: 3

</div>

## Modified z-score trigger

The modified z-score of Iglewicz and Hoaglin (1993) measures how far a peak stands above the median $\tilde{x}$ of the detection function, in units of the MAD scaled to a standard deviation:

$$
z = \frac{x - \tilde{x}}{\mathrm{MAD} / 0.6745}
$$

A detection needs a height of at least the median plus `z_score` scaled MADs, and a prominence of at least `z_score` scaled MADs, 7 by default. Unlike the `MADTrigger`, the threshold rises with the noise floor of the detection function.

!!! abstract "Citation"
    Iglewicz, B., and Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press, Milwaukee.

```python exec='on'
from qseek.utils import json_example
from qseek.triggers import ModZScoreTrigger

print(json_example(ModZScoreTrigger()))
```

<div class="qs-config" markdown>

::: qseek.triggers.ModZScoreTrigger
    options:
      heading_level: 3

</div>

## Fixed threshold

`ThresholdTrigger` applies the same minimum semblance to all windows, for height and prominence. Set it from the semblance of the events you want to detect: the `semblance` column of `csv/detections.csv` of a first run shows it.

```python exec='on'
from qseek.utils import json_example
from qseek.triggers import ThresholdTrigger

print(json_example(ThresholdTrigger()))
```

<div class="qs-config" markdown>

::: qseek.triggers.ThresholdTrigger
    options:
      heading_level: 3

</div>

---
icon: lucide/waves
---

[](){ #qseek.features.FeatureExtractorType }

# Event features

Event features describe the detected earthquakes beyond their location and magnitude. `features` is a list of feature extractors; Qseek runs them for every detection and stores the results with the detections.

## Ground motion

Measures the peak ground acceleration (PGA), the peak horizontal acceleration and the peak ground velocity (PGV) on the restituted waveforms of every station. The [stations](stations.md) need instrument responses from StationXML. The event features are the maxima over all stations.

```python exec='on'
from qseek.utils import json_example
from qseek.features.ground_motion import GroundMotionExtractor

print(json_example(GroundMotionExtractor()))
```

<div class="qs-config" markdown>

::: qseek.features.ground_motion.GroundMotionExtractor
    options:
      heading_level: 3

</div>

To extract the features of an existing run again, see [recalculate magnitudes and features](../guides/manage-runs.md#recalculate-magnitudes-and-features).

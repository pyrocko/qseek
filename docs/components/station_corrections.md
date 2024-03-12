# Station Corrections

Station corrections can be extract from previous runs to refine the localisation accuracy. The corrections can also help to improve the semblance find more events in a dataset.

## Station Specific Corrections

![Source specific delay statistic](../images/station-delay-times.webp)
*Statistics of station delay times.*

```python exec='on'
from qseek.utils import generate_docs
from qseek.insights import StationCorrections

print(generate_docs(StationCorrections()))
```

## Source Specific Corrections

![Source specific corrections volume](../images/delay-volume.webp)
*Delay volume for a selected stations.*

```python exec='on'
from qseek.utils import generate_docs
from qseek.insights import SourceSpecificStationCorrections

print(generate_docs(SourceSpecificStationCorrections()))
```

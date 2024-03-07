# Station Corrections

Station corrections can be extract from previous runs to refine the localisation accuracy. The corrections can also help to improve the semblance find more events in a dataset.

```python exec='on'
from qseek.utils import generate_docs
from qseek.insights import StationCorrections

print(generate_docs(StationCorrections()))
```


```python exec='on'
from qseek.utils import generate_docs
from qseek.insights import SourceSpecificStationCorrections

print(generate_docs(SourceSpecificStationCorrections()))
```

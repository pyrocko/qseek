## Station Metadata

Meta data is required primarily for **station locations and codes**.

Supported data formats are:

* [x] [StationXML](https://www.fdsn.org/xml/station/)
* [x] [Pyrocko Station YAML](https://pyrocko.org/docs/current/formats/yaml.html)

Metadata does not need to include response information for pure detection and localisation. If local magnitudes M~L~ are extracted, response information is required.

```python exec='on'
from qseek.utils import generate_docs
from qseek.models.station import Stations

print(generate_docs(Stations()))
```

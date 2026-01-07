# Station Metadata

Station metadata is required primarily for **station locations and codes**.

Supported data formats are:

* [x] [StationXML](https://www.fdsn.org/xml/station/)
* [x] [Pyrocko Station YAML](https://pyrocko.org/docs/current/formats/yaml.html)

If local magnitudes M~L~ are extracted, response information as StationXML is required.

```python exec='on'
from qseek.utils import generate_docs
from qseek.models.station import StationInventory

print(generate_docs(StationInventory()))
```

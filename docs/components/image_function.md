# Image Function

For image functions this version of Qseek relies heavily on machine learning pickers delivered by [SeisBench](https://github.com/seisbench/seisbench).

## SeisBench Image Function

SeisBench offers access to a variety of machine learning phase pickers pre-trained on various data sets.

!!! abstract "Citation PhaseNet"
    *Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method." arXiv preprint arXiv:1803.03211 (2018).*

```python exec='on'
from qseek.utils import generate_docs
from qseek.images.seisbench import SeisBench

print(generate_docs(SeisBench()))
```

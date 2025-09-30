# Earthquake Magnitude Calculation

Qseek supports earthquake magnitude calculation as Local Magnitudes in different attenuation models and Moment Magnitudes derived from forward-modelled attenuation curves.

## Local Magnitude

Local magnitude calculation relies on a tuned attenuation model.

```python exec='on'
from qseek.utils import generate_docs
from qseek.magnitudes.local_magnitude import LocalMagnitudeExtractor

print(generate_docs(LocalMagnitudeExtractor()))
```

## Moment Magnitude

Based on forward-modelled attenuation curves using [Pyrocko-GF](https://pyrocko.org/docs/current/topics/pyrocko-gf.html). For more information on the method see [Dahm et al., 2024](https://doi.org/10.26443/seismica.v3i2.1205).

```python exec='on'
from qseek.utils import generate_docs
from qseek.magnitudes.moment_magnitude import MomentMagnitudeExtractor

print(generate_docs(MomentMagnitudeExtractor()))
```

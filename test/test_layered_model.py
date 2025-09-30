from qseek.models.layered_model import LayeredModel
from qseek.tracers.utils import EarthModel

KM = 1e3

GRADIENT_MODEL_ND = """    -5.0    1.700  0.950   2.8
     0.0    1.700  0.950   2.8
     0.5    1.838  1.028   2.8
     1.0    2.400  1.344   2.8
     3.0    3.895  2.177   2.8
    10.0    5.624  3.144   2.8
    23.0    6.160  3.457   2.8
mantle
    37.0    8.200  4.600   2.8
"""


def test_layered_model(plot: bool) -> None:
    earth_model = EarthModel()
    layered_model = LayeredModel.from_earth_model(earth_model)

    if plot:
        layered_model.plot(depth_range=(0, 20 * KM), samples=1000)


def test_gradient_model(plot: bool) -> None:
    earth_model = EarthModel(raw_file_data=GRADIENT_MODEL_ND)
    layered_model = LayeredModel.from_earth_model(earth_model)

    if plot:
        layered_model.plot(depth_range=(-5 * KM, 35 * KM), samples=1000)

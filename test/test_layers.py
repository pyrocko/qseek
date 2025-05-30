from qseek.models.layered_model import LayeredModel
from qseek.tracers.utils import EarthModel

KM = 1e3


def test_layered_model():
    earth_model = EarthModel()
    layered_model = LayeredModel.from_earth_model(earth_model)

    if True:
        layered_model.plot(depth_range=(0, 10 * KM), samples=100)

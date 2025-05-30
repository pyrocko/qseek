from qseek.models.layered_model import LayeredModel
from qseek.tracers.utils import EarthModel

KM = 1e3


def test_layered_model() -> None:
    earth_model = EarthModel()
    layered_model = LayeredModel.from_earth_model(earth_model)

    if True:
        layered_model.plot(depth_range=(0, 20 * KM), samples=1000)

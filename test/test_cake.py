from lassie.config import Config
from lassie.tracers import CakeTracer


def test_cake_tracer(sample_config: Config):
    config = sample_config
    tracer = CakeTracer(
        earthmodel="""
   0.00      5.50    3.59    2.7
   1.00      5.50    3.59    2.7
   1.00      6.00    3.92    2.7
   4.00      6.00    3.92    2.7
   4.00      6.20    4.05    2.7
   8.00      6.20    4.05    2.7
   8.00      6.30    4.12    2.7
   13.00     6.30    4.12    2.7
   13.00     6.40    4.18    2.7
   17.00     6.40    4.18    2.7
   17.00     6.50    4.25    2.7
   22.00     6.50    4.25    2.7
   22.00     6.60    4.31    2.7
   26.00     6.60    4.31    2.7
   26.00     6.80    4.44    2.7
   30.00     6.80    4.44    2.7
   30.00     8.10    5.29    2.7
   45.00     8.10    5.29    2.7
   45.00     8.50    5.56    2.7
   71.00     8.50    5.56    2.7
   71.00     8.73    5.71    2.7
   101.00    8.73    5.71    2.7
   101.00    8.73    5.71    2.7
   201.00    8.73    5.71    2.7
   201.00    8.73    5.71    2.7
   301.00    8.73    5.71    2.7
   301.00    8.73    5.71    2.7
   401.00    8.73    5.71    2.7
    """
    )

    traveltime = tracer.get_traveltime(
        "cake:P", node=config.octree[0], station=config.stations.stations[0]
    )
    print(traveltime)

    traveltimes = tracer.get_traveltimes(
        "cake:P", octree=config.octree, stations=config.stations
    )
    print(traveltimes)
    traveltimes = tracer.get_traveltimes(
        "cake:P", octree=config.octree, stations=config.stations
    )

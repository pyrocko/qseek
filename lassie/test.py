from lassie import plot
from lassie.models.detection import Detections

detections = Detections(rundir="test-lassie/")
detection = detections.detections[0]

plot.plot_detection(detection)

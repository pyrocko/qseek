from lassie import plot
from lassie.models.detection import EventDetections

detections = EventDetections(rundir="test-lassie/")
detection = detections.detections[0]

plot.plot_detection(detection)

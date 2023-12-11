from qseek import plot
from qseek.models.detection import EventDetections

detections = EventDetections(rundir="test-qseek/")
detection = detections.detections[0]

plot.plot_detection(detection)

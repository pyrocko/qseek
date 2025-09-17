from qseek import plot
from qseek.models.catalog import EventCatalog

detections = EventCatalog(rundir="test-qseek/")
detection = detections.events[0]

plot.plot_detection(detection)

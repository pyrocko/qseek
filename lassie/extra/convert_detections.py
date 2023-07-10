import argparse
from pathlib import Path
from typing import Any, Iterator

from pydantic import Field
from rich.console import Console
from rich.progress import track

from lassie.features import EventFeaturesTypes
from lassie.models.detection import (
    FILENAME_DETECTIONS,
    FILENAME_RECEIVERS,
    EventDetection,
    EventDetections,
    EventFeatures,
    EventReceivers,
    Receiver,
)

console = Console()


class Detection(EventDetection):
    features: list[EventFeaturesTypes] = []
    old_receivers: list[Receiver] = Field([], alias="receivers")

    def model_post_init(self, __context: Any) -> None:
        ...

    def to_new(self) -> EventDetection:
        detection = EventDetection(**self.model_dump(exclude={"features", "receivers"}))
        detection._features = EventFeatures(
            event_uid=self.uid,
            features=self.features,
        )
        detection._receivers = EventReceivers(
            event_uid=self.uid,
            receivers=self.old_receivers,
        )
        return detection


def iter_detections(rundir: Path) -> Iterator[EventDetection]:
    detection_dir = rundir / "detections"
    files = sorted(detection_dir.glob("*.json"))

    for detection_file in track(
        files, description=f"Converting {len(files)} detections..."
    ):
        yield Detection.model_validate_json(detection_file.read_text()).to_new()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", type=Path)
    args = parser.parse_args()

    for filename in (FILENAME_RECEIVERS, FILENAME_DETECTIONS):
        filename = args.rundir / filename
        if filename.exists():
            filename.unlink()

    new_detections = EventDetections(rundir=args.rundir)

    for detection in iter_detections(args.rundir):
        new_detections.add(detection)

    new_detections.dump_detections(jitter_location=100.0)

    with console.status("Validating new detection format..."):
        detections = EventDetections.load_rundir(rundir=args.rundir)
        console.print(f"Number of detections: {detections.n_detections}")
        for detection in detections.detections:
            print(detection.receivers.event_uid, detection.uid)
            break


if __name__ == "__main__":
    main()

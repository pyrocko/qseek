#!/usr/bin/env python3
"""Convert the HypoDD relocations of a Qseek export to CSV and Pyrocko events.

`qseek export hypodd` copies this script into the HypoDD project folder, `run.sh`
runs it after hypoDD. It reads `hypoDD.reloc` and `event_ids.csv` and writes

- `hypodd_relocations.csv`: one relocated event per row, with the origin time in
  ISO 8601, the location, the HypoDD statistics, the Qseek detection and the shift
  from the Qseek location, and a `WKT_geom` column for QGIS;
- `hypodd_relocations.yaml`: the relocated events as Pyrocko events, if Pyrocko is
  installed.

Run it again by hand in the project folder:

    python3 hypodd_results.py

The script needs only the Python standard library, and Pyrocko for the YAML file.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrocko.model import Event

logger = logging.getLogger("hypodd_results")

EARTH_RADIUS = 6371e3
KM = 1e3
# hypoDD writes -9 for statistics of data types that are not used
NOT_USED = -9.0

RELOC_COLUMNS = (
    "id",
    "lat",
    "lon",
    "depth",
    "x",
    "y",
    "z",
    "ex",
    "ey",
    "ez",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "mag",
    "nccp",
    "nccs",
    "nctp",
    "ncts",
    "rcc",
    "rct",
    "cid",
)

CSV_COLUMNS = (
    "time",
    "lat",
    "lon",
    "depth",
    "magnitude",
    "magnitude_type",
    "uid",
    "hypodd_id",
    "cluster",
    "x",
    "y",
    "z",
    "error_x",
    "error_y",
    "error_z",
    "n_ct_p",
    "n_ct_s",
    "n_cc_p",
    "n_cc_s",
    "rms_ct",
    "rms_cc",
    "qseek_time",
    "qseek_lat",
    "qseek_lon",
    "qseek_depth",
    "shift_east",
    "shift_north",
    "shift_horizontal",
    "shift_depth",
    "shift_time",
    "WKT_geom",
)


@dataclass
class Relocation:
    """A relocated event of hypoDD.reloc and its Qseek detection.

    Depths and shifts are in meters, depths below sea level; times in seconds.
    """

    hypodd_id: int
    time: datetime
    lat: float
    lon: float
    depth: float
    cluster: int
    x: float
    y: float
    z: float
    error_x: float
    error_y: float
    error_z: float
    n_ct_p: int
    n_ct_s: int
    n_cc_p: int
    n_cc_s: int
    rms_ct: float | None
    rms_cc: float | None
    hypodd_magnitude: float
    uid: str = ""
    qseek_time: datetime | None = None
    qseek_lat: float | None = None
    qseek_lon: float | None = None
    qseek_depth: float | None = None
    magnitude: float | None = None
    magnitude_type: str = ""

    def shift(self) -> tuple[float, float, float] | None:
        """Shift east, north and down from the Qseek location in m."""
        if self.qseek_lat is None or self.qseek_lon is None:
            return None
        cos_lat = math.cos(math.radians(self.qseek_lat))
        east = math.radians(self.lon - self.qseek_lon) * cos_lat * EARTH_RADIUS
        north = math.radians(self.lat - self.qseek_lat) * EARTH_RADIUS
        depth = self.depth - self.qseek_depth if self.qseek_depth is not None else 0.0
        return east, north, depth

    def csv_row(self) -> dict[str, str]:
        """Row of the CSV file."""

        def num(value: float | None, fmt: str) -> str:
            return "" if value is None else format(value, fmt)

        shift = self.shift()
        shift_time = (
            (self.time - self.qseek_time).total_seconds() if self.qseek_time else None
        )
        return {
            "time": iso_time(self.time),
            "lat": f"{self.lat:.6f}",
            "lon": f"{self.lon:.6f}",
            "depth": f"{self.depth:.1f}",
            "magnitude": num(self.magnitude, ".2f"),
            "magnitude_type": self.magnitude_type,
            "uid": self.uid,
            "hypodd_id": str(self.hypodd_id),
            "cluster": str(self.cluster),
            "x": f"{self.x:.1f}",
            "y": f"{self.y:.1f}",
            "z": f"{self.z:.1f}",
            "error_x": f"{self.error_x:.1f}",
            "error_y": f"{self.error_y:.1f}",
            "error_z": f"{self.error_z:.1f}",
            "n_ct_p": str(self.n_ct_p),
            "n_ct_s": str(self.n_ct_s),
            "n_cc_p": str(self.n_cc_p),
            "n_cc_s": str(self.n_cc_s),
            "rms_ct": num(self.rms_ct, ".4f"),
            "rms_cc": num(self.rms_cc, ".4f"),
            "qseek_time": iso_time(self.qseek_time) if self.qseek_time else "",
            "qseek_lat": num(self.qseek_lat, ".6f"),
            "qseek_lon": num(self.qseek_lon, ".6f"),
            "qseek_depth": num(self.qseek_depth, ".1f"),
            "shift_east": num(shift[0] if shift else None, ".1f"),
            "shift_north": num(shift[1] if shift else None, ".1f"),
            "shift_horizontal": num(math.hypot(*shift[:2]) if shift else None, ".1f"),
            "shift_depth": num(shift[2] if shift else None, ".1f"),
            "shift_time": num(shift_time, ".3f"),
            "WKT_geom": f"POINT Z({self.lon:.6f} {self.lat:.6f} {-self.depth:.1f})",
        }

    def pyrocko_event(self) -> Event:
        """Get the relocation as Pyrocko event, named by its origin time like Qseek."""
        from pyrocko.model import Event

        extras = {
            "hypodd_id": self.hypodd_id,
            "cluster": self.cluster,
            "n_ct_p": self.n_ct_p,
            "n_ct_s": self.n_ct_s,
            "rms_ct": self.rms_ct,
        }
        if self.uid:
            extras["qseek_uid"] = self.uid
        if self.qseek_time:
            extras["qseek_time"] = iso_time(self.qseek_time)
        return Event(
            name=iso_time(self.time),
            time=self.time.timestamp(),
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
            magnitude=self.magnitude,
            magnitude_type=self.magnitude_type or None,
            extras={key: value for key, value in extras.items() if value is not None},
        )


def iso_time(time: datetime) -> str:
    """ISO 8601 time in UTC with millisecond resolution, e.g. 2024-05-20T00:17:52.540Z.

    Returns:
        str: The formatted time.
    """
    return (
        time.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def read_reloc(file: Path) -> list[Relocation]:
    """Read hypoDD.reloc.

    Returns:
        list[Relocation]: The relocated events in the order of the file.
    """
    relocations = []
    for line in file.read_text().splitlines():
        values = line.split()
        if len(values) != len(RELOC_COLUMNS):
            continue
        row = dict(zip(RELOC_COLUMNS, values, strict=True))
        time = datetime(
            int(row["year"]),
            int(row["month"]),
            int(row["day"]),
            int(row["hour"]),
            int(row["minute"]),
            tzinfo=UTC,
        ) + timedelta(seconds=float(row["second"]))
        rms_ct, rms_cc = float(row["rct"]), float(row["rcc"])
        relocations.append(
            Relocation(
                hypodd_id=int(row["id"]),
                time=time,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                depth=float(row["depth"]) * KM,
                cluster=int(row["cid"]),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                error_x=float(row["ex"]),
                error_y=float(row["ey"]),
                error_z=float(row["ez"]),
                n_ct_p=int(row["nctp"]),
                n_ct_s=int(row["ncts"]),
                n_cc_p=int(row["nccp"]),
                n_cc_s=int(row["nccs"]),
                rms_ct=None if rms_ct == NOT_USED else rms_ct,
                rms_cc=None if rms_cc == NOT_USED else rms_cc,
                hypodd_magnitude=float(row["mag"]),
            )
        )
    return relocations


def add_detections(relocations: list[Relocation], event_ids: Path) -> None:
    """Add the Qseek detections of `event_ids.csv` to the relocations.

    Exports of older Qseek versions list only the UID and the origin time; the
    magnitude then comes from hypoDD.reloc.
    """
    with event_ids.open(newline="") as file:
        detections = {int(row["id"]): row for row in csv.DictReader(file)}
    for relocation in relocations:
        row = detections.get(relocation.hypodd_id)
        if row is None:
            logger.warning("event %d not in %s", relocation.hypodd_id, event_ids)
            continue
        relocation.uid = row.get("uid", "")
        relocation.qseek_time = datetime.fromisoformat(row["time"])
        relocation.qseek_lat = optional_float(row.get("lat"))
        relocation.qseek_lon = optional_float(row.get("lon"))
        relocation.qseek_depth = optional_float(row.get("depth"))
        if "magnitude" in row:
            relocation.magnitude = optional_float(row["magnitude"])
            relocation.magnitude_type = row.get("magnitude_type", "")
        else:
            relocation.magnitude = relocation.hypodd_magnitude


def write_csv(relocations: list[Relocation], file: Path) -> None:
    with file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(relocation.csv_row() for relocation in relocations)


def write_pyrocko_events(relocations: list[Relocation], file: Path) -> bool:
    """Write the relocations as Pyrocko events.

    Returns:
        bool: False if Pyrocko is not installed.
    """
    try:
        from pyrocko.model import dump_events
    except ImportError:
        return False
    events = [relocation.pyrocko_event() for relocation in relocations]
    dump_events(events, str(file), format="yaml")
    return True


def convert(
    directory: Path, output: str = "hypodd_relocations", reloc: str = "hypoDD.reloc"
) -> list[Relocation]:
    """Convert the relocations of a HypoDD project folder.

    Args:
        directory: The HypoDD project folder of `qseek export hypodd`.
        output: Name of the output files, without extension.
        reloc: Name of the hypoDD relocation file.

    Returns:
        list[Relocation]: The relocations, sorted by time.
    """
    reloc_file = directory / reloc
    if not reloc_file.exists():
        raise FileNotFoundError(f"{reloc_file} not found, run hypoDD first")
    relocations = sorted(read_reloc(reloc_file), key=lambda r: r.time)
    event_ids = directory / "event_ids.csv"
    if event_ids.exists():
        add_detections(relocations, event_ids)
        with event_ids.open(newline="") as file:
            n_exported = sum(1 for _ in csv.DictReader(file))
        logger.info("%d of %d exported events relocated", len(relocations), n_exported)
    else:
        logger.warning("%s not found, the Qseek detections are missing", event_ids)

    csv_file = directory / f"{output}.csv"
    write_csv(relocations, csv_file)
    logger.info("wrote %s", csv_file)
    yaml_file = directory / f"{output}.yaml"
    if write_pyrocko_events(relocations, yaml_file):
        logger.info("wrote %s", yaml_file)
    else:
        logger.warning("Pyrocko is not installed, %s not written", yaml_file)
    return relocations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="HypoDD project folder, default: the folder of this script",
    )
    parser.add_argument(
        "--output",
        default="hypodd_relocations",
        help="name of the output files, without extension",
    )
    parser.add_argument(
        "--reloc", default="hypoDD.reloc", help="hypoDD relocation file"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    convert(args.directory, args.output, args.reloc)


if __name__ == "__main__":
    main()

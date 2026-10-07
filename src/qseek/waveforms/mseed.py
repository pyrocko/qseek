"""Read time windows from MiniSEED files without scanning the whole file.

Pyrocko reads a time window by walking over the headers of all records in the
file. For a day file of an SDS archive, that is the whole file for every window.
Most MiniSEED files have records of a fixed length in time order: the records of
a window can then be found by bisection and read in one go. A sparse index of the
start times narrows the bisection down to a block of records, read at once.
"""

from __future__ import annotations

import logging
import os
import struct
from array import array
from bisect import bisect_right
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from pyrocko.io.mseed import iload
from pyrocko.trace import NoData

if TYPE_CHECKING:
    from pyrocko.trace import Trace

logger = logging.getLogger(__name__)

# libmseed and pyrocko's tmin and tmax use microseconds
HPT_PER_SECOND = 1_000_000
# Times in the fixed header are in units of 0.1 ms
HPT_PER_TICK = 100

EPOCH_ORDINAL = date(1970, 1, 1).toordinal()

# Bytes of the fixed header read for each record
FIXED_HEADER_LENGTH = 48
# Start time, number of samples, activity flags and time correction from byte 20
START_TIME_FORMAT = "HHBBBxHH4xB3xi"
QUALITY_CODES = np.frombuffer(b"DRQM", dtype=np.uint8)

# The index holds the start time of every INDEX_STRIDE-th record
INDEX_STRIDE = 16


class RecordLayout(NamedTuple):
    record_length: int
    n_records: int
    byte_order: str
    sampling_rate: float
    index: array[int]


@lru_cache(maxsize=2)
def _header_dtype(byte_order: str) -> np.dtype:
    return np.dtype(
        {
            "names": [
                "year",
                "day_of_year",
                "hour",
                "minute",
                "second",
                "fraction",
                "n_samples",
                "rate_factor",
                "rate_multiplier",
                "activity_flags",
                "time_correction",
            ],
            "formats": [
                byte_order + "u2",
                byte_order + "u2",
                "u1",
                "u1",
                "u1",
                byte_order + "u2",
                byte_order + "u2",
                byte_order + "i2",
                byte_order + "i2",
                "u1",
                byte_order + "i4",
            ],
            "offsets": [20, 22, 24, 25, 26, 28, 30, 32, 34, 36, 40],
            "itemsize": FIXED_HEADER_LENGTH,
        }
    )


def _start_times(headers: np.ndarray, byte_order: str) -> np.ndarray:
    """Start times of the records in microseconds, as libmseed.

    Args:
        headers: Fixed headers of the records, shape (n_records, 48).
        byte_order: Byte order of the records, `>` or `<`.
    """
    values = np.ascontiguousarray(headers).view(_header_dtype(byte_order))[:, 0]
    year = values["year"].astype(np.int64)
    days = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    days = days.astype(np.int64) + values["day_of_year"] - 1
    seconds = (
        days * 86400
        + values["hour"].astype(np.int64) * 3600
        + values["minute"].astype(np.int64) * 60
        + values["second"]
    )
    ticks = seconds * 10_000 + values["fraction"]
    # Bit 1 of the activity flags: time correction applied
    correct_time = (values["activity_flags"] & 0x02) == 0
    ticks += np.where(correct_time, values["time_correction"], 0)
    return ticks * HPT_PER_TICK


def _end_times(
    headers: np.ndarray,
    start_times: np.ndarray,
    byte_order: str,
    sampling_rate: float,
) -> np.ndarray:
    """Times of the last samples of the records in microseconds, as libmseed."""
    values = np.ascontiguousarray(headers).view(_header_dtype(byte_order))[:, 0]
    n_samples = values["n_samples"].astype(np.float64)
    # Truncated towards zero as libmseed
    span = ((n_samples - 1.0) / sampling_rate * HPT_PER_SECOND + 0.5).astype(np.int64)
    return start_times + span


def _sampling_rates(headers: np.ndarray, byte_order: str) -> np.ndarray:
    values = np.ascontiguousarray(headers).view(_header_dtype(byte_order))[:, 0]
    factor = values["rate_factor"].astype(np.float64)
    multiplier = values["rate_multiplier"].astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.select(
            [
                (factor > 0) & (multiplier > 0),
                (factor > 0) & (multiplier < 0),
                (factor < 0) & (multiplier > 0),
                (factor < 0) & (multiplier < 0),
            ],
            [
                factor * multiplier,
                -factor / multiplier,
                -multiplier / factor,
                1.0 / (factor * multiplier),
            ],
            default=0.0,
        )


def _find_blockette_1000(header: bytes) -> tuple[str, int, int] | None:
    """Byte order, record length and offset of blockette 1000 of the first record."""
    if len(header) < FIXED_HEADER_LENGTH:
        return None
    year = struct.unpack_from(">H", header, 20)[0]
    byte_order = ">" if 1900 <= year <= 2100 else "<"

    offset = struct.unpack_from(byte_order + "H", header, 46)[0]
    for _ in range(16):
        if offset < FIXED_HEADER_LENGTH or offset + 8 > len(header):
            return None
        blockette_type, next_offset = struct.unpack_from(
            byte_order + "HH", header, offset
        )
        if blockette_type == 1000:
            return byte_order, 1 << header[offset + 6], offset
        if not next_offset:
            return None
        offset = next_offset
    return None


@lru_cache(maxsize=16384)
def _record_layout(path: str, size: int, mtime_ns: int) -> RecordLayout | None:
    """Layout of a file with records of a fixed length in time order.

    Returns None when the records differ in length, channel or sampling rate, or
    when they overlap or are out of time order. Such files are read by scanning
    all records.
    """
    with open(path, "rb") as file:
        first_header = file.read(512)
    blockette_1000 = _find_blockette_1000(first_header)
    if blockette_1000 is None:
        return None
    byte_order, record_length, blockette_offset = blockette_1000
    if record_length < 128 or not size or size % record_length:
        return None
    n_records = size // record_length

    records = np.memmap(
        path, dtype=np.uint8, mode="r", shape=(n_records, record_length)
    )
    headers = np.array(records[:, : max(blockette_offset + 8, FIXED_HEADER_LENGTH)])
    del records

    if not np.isin(headers[:, 6], QUALITY_CODES).all():
        return None
    # Same network, station, location and channel codes
    if (headers[:, 8:20] != headers[0, 8:20]).any():
        return None
    blockette = headers[:, blockette_offset : blockette_offset + 8]
    if (blockette != blockette[0]).any():
        return None

    headers = headers[:, :FIXED_HEADER_LENGTH]
    year = headers[:, 20:22].copy().view(byte_order + "u2")
    if ((year < 1900) | (year > 2100)).any():
        return None

    rates = _sampling_rates(headers, byte_order)
    if rates[0] <= 0.0 or (rates != rates[0]).any():
        return None
    sampling_rate = float(rates[0])

    # Records in time order, each starting after the last sample of the one before
    start_times = _start_times(headers, byte_order)
    end_times = _end_times(headers, start_times, byte_order, sampling_rate)
    if (start_times[1:] <= end_times[:-1]).any():
        return None

    return RecordLayout(
        record_length=record_length,
        n_records=n_records,
        byte_order=byte_order,
        sampling_rate=sampling_rate,
        index=array("q", start_times[::INDEX_STRIDE].tolist()),
    )


def get_layout(path: Path) -> RecordLayout | None:
    """Layout of a MiniSEED file if its time windows can be read directly, cached."""
    try:
        stat = path.stat()
        return _record_layout(str(path), stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError) as exc:
        logger.debug("cannot index the records of %s: %s", path, exc)
        return None


def _record_times(layout: RecordLayout, data: bytes, offset: int) -> tuple[int, int]:
    """Start time and time of the last sample of a record in microseconds."""
    (
        year,
        day_of_year,
        hour,
        minute,
        second,
        fraction,
        n_samples,
        activity_flags,
        time_correction,
    ) = struct.unpack_from(layout.byte_order + START_TIME_FORMAT, data, offset + 20)
    days = date(year, 1, 1).toordinal() - EPOCH_ORDINAL + day_of_year - 1
    seconds = days * 86400 + hour * 3600 + minute * 60 + second
    ticks = seconds * 10_000 + fraction
    if not activity_flags & 0x02:
        ticks += time_correction
    start_time = ticks * HPT_PER_TICK
    span = int((n_samples - 1) / layout.sampling_rate * HPT_PER_SECOND + 0.5)
    return start_time, start_time + span


def _find_record(fd: int, layout: RecordLayout, time: int) -> tuple[int, int]:
    """Find the last record that starts at or before a time in microseconds.

    Returns:
        tuple[int, int]: Index of the record and time of its last sample, -1 and
            -1 if all records start after the time.
    """
    i_block = bisect_right(layout.index, time) - 1
    if i_block < 0:
        return -1, -1
    first = i_block * INDEX_STRIDE
    n_records = min(INDEX_STRIDE, layout.n_records - first)
    record_length = layout.record_length
    data = os.pread(fd, n_records * record_length, first * record_length)

    # The first record of the block starts at or before the time
    low, high = 1, n_records
    while low < high:
        middle = (low + high) // 2
        if _record_times(layout, data, middle * record_length)[0] <= time:
            low = middle + 1
        else:
            high = middle
    i_record = low - 1
    _, end_time = _record_times(layout, data, i_record * record_length)
    return first + i_record, end_time


def _select_records(fd: int, layout: RecordLayout, tmin: int, tmax: int) -> range:
    """Records overlapping the time window from tmin to tmax, in microseconds.

    The same records as pyrocko selects: from the first record that ends at or after
    tmin, to the last record that starts at or before tmax.
    """
    i_record, end_time = _find_record(fd, layout, tmin)
    # Skip the record holding tmin when it ends before tmin
    first = i_record + 1 if i_record < 0 or end_time < tmin else i_record
    last, _ = _find_record(fd, layout, tmax)
    return range(first, last + 1)


def load_time_window(path: Path, tmin: float, tmax: float) -> list[Trace]:
    """Load the traces of a MiniSEED file between tmin and tmax.

    Returns the same traces as `pyrocko.io.mseed.iload` with `tmin` and `tmax`,
    but reads only the records within the time window when the file has records
    of a fixed length in time order.

    Args:
        path: The MiniSEED file.
        tmin: Start time in seconds since epoch.
        tmax: End time in seconds since epoch, excluded.

    Returns:
        list[Trace]: The traces chopped to the time window.
    """
    layout = get_layout(path)
    if layout is None:
        return list(iload(str(path), tmin=tmin, tmax=tmax))

    fd = os.open(path, os.O_RDONLY)
    try:
        records = _select_records(
            fd,
            layout,
            int(tmin * HPT_PER_SECOND),
            int(tmax * HPT_PER_SECOND),
        )
    finally:
        os.close(fd)
    if not records:
        return []

    traces = []
    for tr in iload(
        str(path),
        offset=records.start * layout.record_length,
        segment_size=len(records) * layout.record_length,
        nsegments=1,
    ):
        try:
            tr.chop(tmin, tmax, include_last=False)
        except NoData:
            continue
        traces.append(tr)
    return traces

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pyrocko.trace import Trace

from qseek.images.base import WaveformImage, WaveformImages
from qseek.images.seisbench import AnnotationPicker
from qseek.models.location import Location
from qseek.models.station import Station, StationInventory
from qseek.octree import Octree
from qseek.search import OctreeSearch
from qseek.station_weights import (
    DistanceWeights,
    LogLogisticWeights,
    StationDensityWeights,
)
from qseek.tracers.constant_velocity import ConstantVelocityTracer
from qseek.tracers.tracers import RayTracers
from qseek.triggers import ThresholdTrigger
from qseek.utils import Range

KM = 1e3
SAMPLING_RATE = 100.0
VELOCITIES = {"constant:P": 5.0 * KM, "constant:S": 2.9 * KM}
PEAK_WIDTH = 0.1

START_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
DURATION = timedelta(seconds=30)
PADDING = timedelta(seconds=10)
EVENT_TIME = START_TIME + timedelta(seconds=12.34)


@pytest.fixture
def small_octree() -> Octree:
    return Octree(
        location=Location(lat=10.0, lon=10.0),
        root_node_size=2 * KM,
        n_levels=4,
        east_bounds=Range(-6 * KM, 6 * KM),
        north_bounds=Range(-6 * KM, 6 * KM),
        depth_bounds=Range(0 * KM, 8 * KM),
    )


@pytest.fixture
def surface_stations() -> StationInventory:
    rng = np.random.default_rng(42)
    return StationInventory(
        stations=[
            Station(
                network="XX",
                station=f"S{idx:02d}",
                lat=10.0,
                lon=10.0,
                north_shift=rng.uniform(-8, 8) * KM,
                east_shift=rng.uniform(-8, 8) * KM,
            )
            for idx in range(12)
        ]
    )


def synthetic_images(
    stations: StationInventory, source: Location, event_time: datetime
) -> WaveformImages:
    """Gaussian image peaks at the true P and S arrivals of the source."""
    tmin = (START_TIME - PADDING).timestamp()
    n_samples = round((DURATION + 2 * PADDING).total_seconds() * SAMPLING_RATE)
    times = tmin + np.arange(n_samples) / SAMPLING_RATE

    images = WaveformImages(start_time=START_TIME, end_time=START_TIME + DURATION)
    for phase, velocity in VELOCITIES.items():
        traces = []
        for station in stations:
            arrival = event_time.timestamp() + source.distance_to(station) / velocity
            traces.append(
                Trace(
                    network=station.network,
                    station=station.station,
                    location=station.location,
                    channel=phase[-1],
                    tmin=tmin,
                    deltat=1.0 / SAMPLING_RATE,
                    ydata=np.exp(-(((times - arrival) / PEAK_WIDTH) ** 2)),
                )
            )
        images.add_image(WaveformImage("Synthetic", phase, 1.0, traces, PEAK_WIDTH))
    images.set_stations(stations)
    return images


@pytest.mark.asyncio
async def test_search_and_pick(
    small_octree: Octree, surface_stations: StationInventory
) -> None:
    surface_stations.prepare(small_octree.location)
    source = Location(
        lat=10.0,
        lon=10.0,
        north_shift=1.3 * KM,
        east_shift=-0.7 * KM,
        depth=3.2 * KM,
    )
    images = synthetic_images(surface_stations, source, EVENT_TIME)

    search = OctreeSearch(
        ray_tracers=RayTracers(
            root=[
                ConstantVelocityTracer(phase=phase, velocity=velocity)
                for phase, velocity in VELOCITIES.items()
            ]
        ),
        window_padding=PADDING,
        trigger=ThresholdTrigger(threshold=1.0),
        ignore_boundary=False,
    )
    detections, semblance = await search.search(images, octree=small_octree)

    assert semblance.ydata.size > 0
    assert len(detections) == 1
    (detection,) = detections
    assert abs((detection.time - EVENT_TIME).total_seconds()) < 0.05
    assert detection.distance_to(source) < 0.5 * KM

    AnnotationPicker(threshold_p=0.5, threshold_s=0.5).add_picks(detections, images)

    receivers = list(detection.receivers)
    assert len(receivers) == surface_stations.n_stations
    for receiver in receivers:
        assert set(receiver.phase_arrivals) == set(VELOCITIES)
        for phase, arrival in receiver.phase_arrivals.items():
            assert arrival.observed is not None, (receiver.nsl, phase)
            expected = EVENT_TIME + timedelta(
                seconds=source.distance_to(receiver) / VELOCITIES[phase]
            )
            assert abs((arrival.observed.time - expected).total_seconds()) <= 0.01
            assert arrival.observed.detection_value == pytest.approx(1.0, abs=0.01)
    assert detection.n_picks == 2 * surface_stations.n_stations


@pytest.mark.asyncio
async def test_search_no_event(
    small_octree: Octree, surface_stations: StationInventory
) -> None:
    surface_stations.prepare(small_octree.location)
    source = Location(lat=10.0, lon=10.0, depth=3 * KM)
    # The arrivals are outside of the padded images
    images = synthetic_images(
        surface_stations, source, START_TIME + DURATION + 3 * PADDING
    )
    search = OctreeSearch(
        ray_tracers=RayTracers(
            root=[
                ConstantVelocityTracer(phase=phase, velocity=velocity)
                for phase, velocity in VELOCITIES.items()
            ]
        ),
        window_padding=PADDING,
        trigger=ThresholdTrigger(threshold=1.0),
        ignore_boundary=False,
    )
    detections, _ = await search.search(images, octree=small_octree)
    assert detections == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "station_weights",
    [DistanceWeights(), StationDensityWeights(), LogLogisticWeights()],
)
async def test_search_station_weights(
    station_weights, small_octree: Octree, surface_stations: StationInventory
) -> None:
    # A dense cluster of stations next to the sparse network
    rng = np.random.default_rng(7)
    stations = StationInventory(
        stations=[
            *surface_stations,
            *(
                Station(
                    network="XX",
                    station=f"C{idx:02d}",
                    lat=10.0,
                    lon=10.0,
                    north_shift=4 * KM + rng.normal(0, 200),
                    east_shift=4 * KM + rng.normal(0, 200),
                )
                for idx in range(8)
            ),
        ]
    )
    stations.prepare(small_octree.location)
    station_weights = station_weights.model_copy(deep=True)
    station_weights.prepare(stations, small_octree)

    source = Location(
        lat=10.0,
        lon=10.0,
        north_shift=-2.1 * KM,
        east_shift=1.6 * KM,
        depth=2.8 * KM,
    )
    images = synthetic_images(stations, source, EVENT_TIME)
    search = OctreeSearch(
        ray_tracers=RayTracers(
            root=[
                ConstantVelocityTracer(phase=phase, velocity=velocity)
                for phase, velocity in VELOCITIES.items()
            ]
        ),
        station_weights=station_weights,
        window_padding=PADDING,
        trigger=ThresholdTrigger(threshold=0.5),
        ignore_boundary=False,
    )
    detections, _ = await search.search(images, octree=small_octree)

    assert len(detections) == 1
    (detection,) = detections
    assert abs((detection.time - EVENT_TIME).total_seconds()) < 0.05
    assert detection.distance_to(source) < 0.5 * KM

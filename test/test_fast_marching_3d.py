from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pytest
import pytest_asyncio
from pyrocko import orthodrome as od

from qseek.models.location import Location
from qseek.models.station import Station, StationInventory
from qseek.octree import Octree
from qseek.tracers.cake import (
    CakeTracer,
    Timing,
)
from qseek.tracers.fast_marching_3d.fast_marching_3d import (
    FastMarching3DTracer,
    InterpolationMethod,
    StationTravelTimeVolume,
)
from qseek.tracers.fast_marching_3d.velocity_models import (
    Constant3DVelocityModel,
    NonLinLocVelocityModel,
    VelocityModel3D,
    VelocityModelLayered,
    grid_offsets,
    node_grid_offsets,
)
from qseek.tracers.utils import DEFAULT_VELOCITY_MODEL_FILE, LayeredEarthModel1D
from qseek.utils import Range, datetime_now

CONSTANT_VELOCITY = 5000
KM = 1e3


class TestVelocityModel(NamedTuple):
    name: str
    wave_type: Literal["P", "S"]
    header_file: str
    reference_location: Location

    def get_header_file(self, data_dir: Path) -> Path:
        return data_dir / self.header_file


TEST_MODELS = [
    TestVelocityModel(
        name="FORGE",
        wave_type=wave_type,
        header_file=f"FORGE_3D_5_large.{wave_type}.mod.hdr",
        reference_location=Location(
            lat=38.50402147,
            lon=-112.8963897,
            elevation=1650.0,
        ),
    )
    for wave_type in ("P", "S")
]


def stations_inside(
    model: VelocityModel3D,
    nstations: int = 20,
    depth_range: float | None = None,
    seed: int = 0,
) -> StationInventory:
    stations = []
    rng = np.random.RandomState(seed)
    for i_sta in range(nstations):
        station = Station(
            network="FM",
            station="STA%02d" % i_sta,
            lat=model.center.lat,
            lon=model.center.lon,
            elevation=model.center.elevation,
            north_shift=model.center.north_shift + rng.uniform(*model.north_bounds),
            east_shift=model.center.east_shift + rng.uniform(*model.east_bounds),
            depth=model.center.depth
            + (
                rng.uniform(depth_range)
                if depth_range is not None
                else rng.uniform(*model.depth_bounds)
            ),
        )
        station = station.shifted_origin()
        stations.append(station)
    return StationInventory(stations=stations)


def octree_cover(model: VelocityModel3D) -> Octree:
    east_excess = KM - model.east_size % (1 * KM)
    north_excess = KM - model.north_size % (1 * KM)
    depth_excess = KM - model.depth_size % (1 * KM)
    return Octree(
        location=model.center,
        root_node_size=1 * KM,
        n_levels=2,
        east_bounds=Range(model.east_bounds[0], model.east_bounds[1] + east_excess),
        north_bounds=Range(model.north_bounds[0], model.north_bounds[1] + north_excess),
        depth_bounds=Range(model.depth_bounds[0], model.depth_bounds[1] + depth_excess),
    )


@pytest_asyncio.fixture
async def station_travel_times(octree: Octree) -> StationTravelTimeVolume:
    octree.location.elevation = 1 * KM
    model = Constant3DVelocityModel(velocity=CONSTANT_VELOCITY, grid_spacing=100.0)
    model_3d = model.get_model(octree, None)
    stations = stations_inside(model_3d)
    return await StationTravelTimeVolume.calculate_from_eikonal(
        model_3d, stations.stations[0]
    )


@pytest.mark.asyncio
async def test_load_save(
    station_travel_times: StationTravelTimeVolume,
    tmp_path: str,
    stations: StationInventory,
) -> None:
    outfile = Path(tmp_path) / "test_fast_marching_station.3dtt"
    station_travel_times.save(outfile)
    assert outfile.exists()

    station_travel_times2 = StationTravelTimeVolume.load(outfile)
    np.testing.assert_equal(
        station_travel_times.travel_times, station_travel_times2.travel_times
    )


@pytest.mark.asyncio
async def test_travel_times_3d_constant(
    station_travel_times: StationTravelTimeVolume,
    octree: Octree,
) -> None:
    eikonal_travel_times = []
    source_distances = []
    for node in octree:
        source = node.as_location()
        eikonal_travel_times.append(
            station_travel_times.interpolate_travel_time(source)
        )
        source_distances.append(station_travel_times.station.distance_to(source))

    eikonal_travel_times = np.array(eikonal_travel_times)
    assert np.any(eikonal_travel_times)

    analytical_travel_times = np.array(source_distances) / CONSTANT_VELOCITY
    nan_travel_times = np.isnan(eikonal_travel_times)

    assert np.any(~nan_travel_times)
    np.testing.assert_almost_equal(
        eikonal_travel_times[~nan_travel_times],
        analytical_travel_times[~nan_travel_times],
        decimal=1,
    )

    eikonal_travel_times = await station_travel_times.interpolate_nodes(
        octree, method="cubic"
    )

    nan_travel_times = np.isnan(eikonal_travel_times)
    assert np.any(~nan_travel_times)
    np.testing.assert_almost_equal(
        eikonal_travel_times[~nan_travel_times],
        analytical_travel_times[~nan_travel_times],
        decimal=1,
    )


@pytest.mark.asyncio
async def test_travel_times_1d_cake(
    octree: Octree,
    fixed_stations: StationInventory,
):
    tracer = FastMarching3DTracer(
        phase="fm:P",
        velocity_model=VelocityModelLayered(
            grid_spacing=200.0,
            velocity="vp",
            filename=DEFAULT_VELOCITY_MODEL_FILE,
        ),
    )
    await tracer.prepare(octree, fixed_stations)

    cake_tracer = CakeTracer(
        phases={"cake:P": Timing(definition="P,p")},
        earthmodel=LayeredEarthModel1D(
            filename=DEFAULT_VELOCITY_MODEL_FILE,
        ),
    )
    await cake_tracer.prepare(octree, fixed_stations)

    travel_times_fast_marching = await tracer.get_travel_times(
        "fm:P", octree, fixed_stations
    )
    travel_times_cake = await cake_tracer.get_travel_times(
        "cake:P", octree, fixed_stations
    )

    nan_mask = np.isnan(travel_times_cake)
    travel_times_fast_marching[nan_mask] = np.nan
    np.testing.assert_almost_equal(
        travel_times_fast_marching, travel_times_cake, decimal=1
    )


@pytest.mark.asyncio
async def test_fast_marching_phase_tracer(
    octree: Octree,
    fixed_stations: StationInventory,
) -> None:
    tracer = FastMarching3DTracer(
        phase="fm:P",
        velocity_model=Constant3DVelocityModel(
            velocity=CONSTANT_VELOCITY, grid_spacing=80.0
        ),
    )
    await tracer.prepare(octree, fixed_stations)
    await tracer.get_travel_times("fm:P", octree, fixed_stations)


@pytest.mark.asyncio
@pytest.mark.parametrize("test_model", TEST_MODELS)
async def test_non_lin_loc(
    data_dir: Path,
    octree: Octree,
    test_model: TestVelocityModel,
) -> None:
    tracer = FastMarching3DTracer(
        phase="fm:P",
        velocity_model=NonLinLocVelocityModel(
            header_file=test_model.get_header_file(data_dir),
            reference_location=test_model.reference_location,
        ).prepare(),
    )
    stations = stations_inside(tracer.velocity_model.get_model(octree, None))
    octree = octree_cover(tracer.velocity_model.get_model(octree, None))
    await tracer.prepare(octree, stations)
    source = octree[1].as_location()
    tracer.get_arrivals(
        "fm:P",
        event_time=datetime_now(),
        source=source,
        receivers=list(stations),
    )


@pytest.mark.plot
@pytest.mark.parametrize("test_model", TEST_MODELS)
def test_nll_model_plot(
    data_dir: Path,
    octree: Octree,
    stations: StationInventory,
    test_model: TestVelocityModel,
) -> None:
    import matplotlib.pyplot as plt

    model = NonLinLocVelocityModel(
        header_file=test_model.get_header_file(data_dir),
        reference_location=test_model.reference_location,
    ).prepare()
    velocity_model = model.get_model(octree, stations).resample(
        grid_spacing=200.0,
        method="linear",
    )

    # 3d figure of velocity model
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    coords = velocity_model.get_meshgrid()
    cmap = ax.scatter(
        coords[0],
        coords[1],
        -coords[2],
        s=np.log(velocity_model.velocity_model.ravel() / KM),
        c=velocity_model.velocity_model.ravel(),
    )
    fig.colorbar(cmap)
    plt.show()


@pytest.mark.plot
@pytest.mark.asyncio
@pytest.mark.parametrize("test_model", TEST_MODELS)
async def test_3d_model_plot(
    data_dir: Path,
    octree: Octree,
    test_model: TestVelocityModel,
) -> None:
    import matplotlib.pyplot as plt

    tracer = FastMarching3DTracer(
        phase="fm:P",
        velocity_model=NonLinLocVelocityModel(
            reference_location=test_model.reference_location,
            header_file=test_model.get_header_file(data_dir),
            grid_spacing=100.0,
        ).prepare(),
    )
    model_3d = tracer.velocity_model.get_model(octree, None)
    octree = octree_cover(model_3d)
    stations = stations_inside(model_3d, depth_range=500.0)
    await tracer.prepare(octree, stations)

    volume = tracer.get_travel_time_volume(stations.stations[3])

    # 3d figure of velocity model
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    coords = volume.get_meshgrid()

    cmap = ax.scatter(
        coords[0],
        coords[1],
        coords[2],
        c=volume.travel_times.ravel(),
        alpha=0.2,
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Depth (m)")
    fig.colorbar(cmap)

    station_offet = volume.station.offset_from(volume.center)
    ax.scatter(*station_offet, s=1000, c="r")
    plt.show()


@pytest.mark.parametrize("test_model", TEST_MODELS)
def test_nll_geometry(
    data_dir: Path,
    tmp_path: Path,
    octree: Octree,
    test_model: TestVelocityModel,
):
    tracer = FastMarching3DTracer(
        velocity_model=NonLinLocVelocityModel(
            reference_location=test_model.reference_location,
            header_file=test_model.get_header_file(data_dir),
            grid_spacing=100.0,
        ).prepare(),
    )

    model_3d = tracer.velocity_model.get_model(octree, None)
    octree = octree_cover(model_3d)

    center = model_3d.center.model_copy()

    size_ns = model_3d.north_bounds[1] - model_3d.north_bounds[0]
    size_ew = model_3d.east_bounds[1] - model_3d.east_bounds[0]

    corner_se = center.model_copy()
    corner_se.east_shift -= size_ew / 2
    corner_se.north_shift -= size_ns / 2

    corner_sw = center.model_copy()
    corner_sw.east_shift += size_ew / 2
    corner_sw.north_shift -= size_ns / 2

    corner_ne = center.model_copy()
    corner_ne.east_shift -= size_ew / 2
    corner_ne.north_shift += size_ns / 2

    corner_nw = center.model_copy()
    corner_nw.east_shift += size_ew / 2
    corner_nw.north_shift += size_ns / 2

    locations = {
        "center": center,
        "reference": test_model.reference_location,
        "corner_se": corner_se,
        "corner_sw": corner_sw,
        "corner_ne": corner_ne,
        "corner_nw": corner_nw,
    }

    # write csv
    csv_file = tmp_path / "corners_model.csv"
    with csv_file.open("w") as f:
        f.write("name, lon, lat\n")
        for name, loc in locations.items():
            f.write(f"{name}, {loc.effective_lon}, {loc.effective_lat}\n")


@pytest.fixture
def constant_model(octree: Octree) -> VelocityModel3D:
    model = Constant3DVelocityModel(velocity=CONSTANT_VELOCITY, grid_spacing=200.0)
    return model.get_model(octree, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("implementation", "rtol"),
    [("eikonalfm", 0.0)],
)
async def test_eikonal_implementations(
    constant_model: VelocityModel3D, implementation: str, rtol: float
) -> None:
    station = stations_inside(constant_model, nstations=1).stations[0]
    volume = await StationTravelTimeVolume.calculate_from_eikonal(
        constant_model, station, implementation=implementation
    )
    assert volume.fmm_tracer == implementation
    distances = constant_model.get_source_distance_grid(station)
    np.testing.assert_allclose(
        volume.travel_times, distances / CONSTANT_VELOCITY, atol=0.05, rtol=rtol
    )


@pytest.mark.asyncio
async def test_eikonal_errors(constant_model: VelocityModel3D) -> None:
    station = stations_inside(constant_model, nstations=1).stations[0]
    with pytest.raises(ValueError, match="unknown eikonal solver"):
        await StationTravelTimeVolume.calculate_from_eikonal(
            constant_model,
            station,
            implementation="unknown",  # type: ignore[arg-type]
        )

    outside = station.model_copy(update={"east_shift": 100 * KM})
    with pytest.raises(ValueError, match="outside"):
        await StationTravelTimeVolume.calculate_from_eikonal(constant_model, outside)


@pytest.mark.asyncio
async def test_tracer_cache_and_vtk(
    octree: Octree,
    fixed_stations: StationInventory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qseek.tracers.fast_marching_3d import fast_marching_3d

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(fast_marching_3d, "FMM_CACHE_DIR", cache_dir)
    stations = StationInventory(stations=fixed_stations.stations[:3])
    n_stations = len(stations.stations)

    def new_tracer() -> FastMarching3DTracer:
        return FastMarching3DTracer(
            phase="fm:P",
            implementation="eikonalfm",
            nthreads=2,
            velocity_model=Constant3DVelocityModel(
                velocity=CONSTANT_VELOCITY, grid_spacing=500.0
            ),
        )

    tracer = new_tracer()
    rundir = tmp_path / "rundir"
    await tracer.prepare(octree, stations, rundir=rundir)
    assert tracer.get_available_phases() == ("fm:P",)
    files = list(cache_dir.glob("*/*/*.3dtt"))
    assert len(files) == n_stations
    vtk_files = {file.name for file in (rundir / "3d-models").iterdir()}
    assert "velocity-model-fm-P.vtr" in vtk_files
    assert len(vtk_files) == n_stations + 1

    # A broken cache file is removed and its travel times recalculated
    files[0].write_bytes(b"broken")
    tracer = new_tracer()
    await tracer.prepare(octree, stations)
    assert len(list(cache_dir.glob("*/*/*.3dtt"))) == n_stations
    assert all(tracer.has_travel_times_volume(station) for station in stations)

    source = octree[0].as_location()
    with pytest.raises(ValueError, match="not supported"):
        await tracer.get_travel_times("fm:S", octree, stations)
    with pytest.raises(ValueError, match="not supported"):
        tracer.get_arrivals("fm:S", datetime_now(), source, list(stations))
    with pytest.raises(ValueError, match="not supported"):
        tracer.get_travel_time_location("fm:S", source, stations.stations[0])


def test_travel_time_volume_errors(
    constant_model: VelocityModel3D, tmp_path: Path
) -> None:
    station = stations_inside(constant_model, nstations=1).stations[0]
    volume = StationTravelTimeVolume(
        center=constant_model.center,
        velocity_model_hash=constant_model.hash(),
        station=station,
        east_bounds=constant_model.east_bounds,
        north_bounds=constant_model.north_bounds,
        depth_bounds=constant_model.depth_bounds,
        grid_spacing=constant_model.grid_spacing,
        fmm_tracer="eikonalfm",
    )
    assert not volume.has_travel_times()
    with pytest.raises(AttributeError, match="not been calculated"):
        volume.save(tmp_path)
    with pytest.raises(FileNotFoundError):
        volume._load_travel_times()


def test_grid_offsets_match_node_geography() -> None:
    center = Location(
        lat=42.7, lon=13.0, elevation=200.0, east_shift=1500.0, north_shift=-700.0
    )
    rng = np.random.default_rng(7)
    stations = [
        Station(
            network="XX",
            station=f"S{i:02d}",
            lat=42.7 + rng.uniform(-0.4, 0.4),
            lon=13.0 + rng.uniform(-0.5, 0.5),
            elevation=rng.uniform(0.0, 1500.0),
        )
        for i in range(20)
    ]
    offsets = grid_offsets(stations, center)
    # Bearing at the center, distance on the ellipsoid
    lat_lon = np.array([sta.effective_lat_lon for sta in stations])
    north, east = od.latlon_to_ne_numpy(
        center.lat, center.lon, lat_lon[:, 0], lat_lon[:, 1]
    )
    np.testing.assert_allclose(offsets[:, 0], east - center.east_shift, atol=1e-6)
    np.testing.assert_allclose(offsets[:, 1], north - center.north_shift, atol=1e-6)
    distance = np.hypot(
        offsets[:, 0] + center.east_shift, offsets[:, 1] + center.north_shift
    )
    geodesic = od.distance_accurate50m_numpy(
        center.lat, center.lon, lat_lon[:, 0], lat_lon[:, 1]
    )
    np.testing.assert_allclose(distance, geodesic, rtol=1e-9)

    # The bearing at the station, reversed, is far off here
    back_north, back_east = od.latlon_to_ne_numpy(
        lat_lon[:, 0], lat_lon[:, 1], center.lat, center.lon
    )
    reversed_offsets = np.column_stack(
        (-back_east - center.east_shift, -back_north - center.north_shift)
    )
    assert np.abs(reversed_offsets - offsets[:, :2]).max() > 50.0

    np.testing.assert_allclose(
        offsets[:, 2],
        [sta.effective_depth - center.effective_depth for sta in stations],
    )

    octree = Octree(
        location=Location(lat=42.75, lon=13.1, elevation=500.0),
        root_node_size=2 * KM,
        n_levels=2,
        east_bounds=Range(-10 * KM, 10 * KM),
        north_bounds=Range(-10 * KM, 10 * KM),
        depth_bounds=Range(0, 4 * KM),
    )
    nodes = list(octree.leaf_nodes)
    for grid_center in (center, octree.location):
        np.testing.assert_allclose(
            node_grid_offsets(nodes, grid_center),
            grid_offsets([node.as_location() for node in nodes], grid_center),
            atol=1e-6,
        )


def cone_volume(
    grid_spacing: float = 500.0,
) -> tuple[StationTravelTimeVolume, np.ndarray]:
    source = np.array([123.0, -77.0, 1000.0])
    volume = StationTravelTimeVolume(
        center=Location(lat=0.0, lon=0.0),
        station=Station(network="XX", station="CONE", lat=0.0, lon=0.0),
        velocity_model_hash="cone",
        east_bounds=Range(-20 * KM, 20 * KM),
        north_bounds=Range(-20 * KM, 20 * KM),
        depth_bounds=Range(0.0, 20 * KM),
        grid_spacing=grid_spacing,
        fmm_tracer="eikonalfm",
    )
    east, north, depth = volume.get_meshgrid()
    distance = np.sqrt(
        (east - source[0]) ** 2 + (north - source[1]) ** 2 + (depth - source[2]) ** 2
    )
    volume._travel_times = (distance / CONSTANT_VELOCITY).astype(np.float32)
    return volume, source


def test_travel_time_volume_interpolation() -> None:
    volume, source = cone_volume()
    rng = np.random.default_rng(1)
    points = rng.uniform(
        [-20 * KM, -20 * KM, 0.0], [20 * KM, 20 * KM, 20 * KM], (2000, 3)
    )
    analytical = np.linalg.norm(points - source, axis=1) / CONSTANT_VELOCITY

    linear = volume.interpolate(points, method="linear")
    cubic = volume.interpolate(points, method="cubic")
    np.testing.assert_allclose(linear, analytical, atol=0.015)
    # The cubic spline is precomputed once; its boundary must keep the gradient
    np.testing.assert_allclose(cubic, analytical, atol=0.002)

    outside = np.array(
        [[-20.01 * KM, 0.0, 0.0], [0.0, 0.0, 20.01 * KM], [0.0, 0.0, -1.0]]
    )
    for method in ("nearest", "linear", "cubic"):
        assert np.isnan(volume.interpolate(outside, method=method)).all()
    edge = np.array([[20 * KM, 20 * KM, 20 * KM]])
    assert np.isfinite(volume.interpolate(edge, method="linear")).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["linear", "cubic"])
async def test_tracer_lut_interpolation(method: InterpolationMethod) -> None:
    location = Location(lat=38.5, lon=-112.9, elevation=500.0)
    octree = Octree(
        location=location,
        root_node_size=1 * KM,
        n_levels=2,
        east_bounds=Range(-3 * KM, 3 * KM),
        north_bounds=Range(-3 * KM, 3 * KM),
        depth_bounds=Range(0, 3 * KM),
    )
    model = Constant3DVelocityModel(velocity=CONSTANT_VELOCITY)
    stations = stations_inside(model.get_model(octree, None), nstations=3)
    tracer = FastMarching3DTracer(
        phase="fm3d:P", velocity_model=model, interpolation_method=method
    )
    await tracer.prepare(octree, stations, use_cached=False)

    nodes = list(octree.leaf_nodes)
    station_list = list(stations)
    lut = await tracer.get_travel_times("fm3d:P", nodes, station_list)
    for i_sta, station in enumerate(station_list):
        volume = tracer.get_travel_time_volume(station)
        expected = volume.interpolate(node_grid_offsets(nodes, volume.center), method)
        np.testing.assert_allclose(lut[:, i_sta], expected, rtol=1e-6)

    outside = Location(lat=38.5, lon=-112.9, east_shift=10 * KM, depth=1 * KM)
    arrivals = tracer.get_arrivals("fm3d:P", datetime_now(), outside, station_list)
    assert arrivals == [None] * len(station_list)


def test_velocity_model_grid_spacing() -> None:
    center = Location(lat=0.0, lon=0.0)
    with pytest.raises(ValueError, match="not a multiple of the grid spacing"):
        VelocityModel3D(
            center=center,
            grid_spacing=312.5,
            east_bounds=(0.0, 20500.0),
            north_bounds=(0.0, 20000.0),
            depth_bounds=(0.0, 10000.0),
        )

    model = VelocityModel3D(
        center=center,
        grid_spacing=500.0,
        east_bounds=(0.0, 20500.0),
        north_bounds=(0.0, 20500.0),
        depth_bounds=(0.0, 10000.0),
    )
    model.set_velocity_model(np.full(model.velocity_model.shape, 5000.0))
    resampled = model.resample(312.5)
    for coords in (
        resampled.east_coords,
        resampled.north_coords,
        resampled.depth_coords,
    ):
        np.testing.assert_allclose(np.diff(coords), resampled.grid_spacing)
    assert resampled.east_bounds[1] <= model.east_bounds[1]


def test_velocity_model_hash_geometry() -> None:
    def model(root_node_size: float) -> VelocityModel3D:
        octree = Octree(
            location=Location(lat=38.5, lon=-112.9),
            root_node_size=root_node_size,
            n_levels=3,
            east_bounds=Range(-5 * root_node_size, 5 * root_node_size),
            north_bounds=Range(-5 * root_node_size, 5 * root_node_size),
            depth_bounds=Range(0, 5 * root_node_size),
        )
        return Constant3DVelocityModel(velocity=CONSTANT_VELOCITY).get_model(
            octree, None
        )

    small, large = model(1 * KM), model(2 * KM)
    assert small.velocity_model.shape == large.velocity_model.shape
    assert small.hash() != large.hash()


def test_velocity_model_layered_depth() -> None:
    octree = Octree(
        location=Location(lat=38.5, lon=-112.9, elevation=1 * KM),
        root_node_size=2 * KM,
        n_levels=3,
        east_bounds=Range(-4 * KM, 4 * KM),
        north_bounds=Range(-4 * KM, 4 * KM),
        depth_bounds=Range(0, 6 * KM),
    )
    layered = VelocityModelLayered(filename=DEFAULT_VELOCITY_MODEL_FILE).prepare()
    model = layered.get_model(octree, None)
    absolute_depths = model.center.effective_depth + model.depth_coords
    expected = [
        layered._layered_model.material(z=depth).vp for depth in absolute_depths
    ]
    np.testing.assert_allclose(model.velocity_model[0, 0, :], expected)


def test_velocity_model_check_velocities() -> None:
    model = VelocityModel3D(
        center=Location(lat=0.0, lon=0.0),
        grid_spacing=100.0,
        east_bounds=(0.0, 1000.0),
        north_bounds=(0.0, 1000.0),
        depth_bounds=(0.0, 1000.0),
    )
    velocities = np.full(model.velocity_model.shape, 5000.0)
    model.set_velocity_model(velocities)
    model.check_velocities()

    velocities[3, 4, 5] = 0.0
    velocities[2, 2, 2] = np.inf
    model.set_velocity_model(velocities)
    with pytest.raises(ValueError, match="2 cells"):
        model.check_velocities()


def test_nll_header_frame(tmp_path: Path) -> None:
    from qseek.tracers.fast_marching_3d.velocity_models import (
        NonLinLocHeader,
    )

    first = "11 7 5  -1.000 -0.500 -0.200  0.1 0.1 0.1 VELOCITY FLOAT\n"
    reference = Location(lat=38.5, lon=-112.9, elevation=1650.0)
    header = tmp_path / "model.P.mod.hdr"

    header.write_text(first)
    with pytest.raises(ValueError, match="reference_location"):
        NonLinLocHeader.from_header_file(header)
    nll = NonLinLocHeader.from_header_file(header, reference_location=reference)
    assert nll.origin.east_shift == pytest.approx(-1000.0)
    assert nll.origin.elevation == pytest.approx(1850.0)

    header.write_text(first + "TRANSFORM  GLOBAL\n")
    with pytest.raises(ValueError, match="GLOBAL"):
        NonLinLocHeader.from_header_file(header, reference_location=reference)

    header.write_text(
        first + "TRANSFORM  SIMPLE LatOrig 38.5 LongOrig -112.9 RotCW 15.0\n"
    )
    with pytest.raises(ValueError, match="rotated"):
        NonLinLocHeader.from_header_file(header, reference_location=reference)

    header.write_text(
        first + "TRANSFORM  SIMPLE LatOrig 38.5 LongOrig -112.9 RotCW 0.0\n"
    )
    NonLinLocHeader.from_header_file(header, reference_location=reference)


def test_velocity_model_is_inside_edge() -> None:
    center = Location(lat=38.5, lon=-112.9)
    model = VelocityModel3D(
        center=center,
        grid_spacing=100.0,
        east_bounds=(-1000.0, 1000.0),
        north_bounds=(-1000.0, 1000.0),
        depth_bounds=(0.0, 1000.0),
    )
    for east, inside in ((1000.0, True), (1000.0 + 1e-6, True), (1000.1, False)):
        location = Location(lat=38.5, lon=-112.9, east_shift=east, depth=500.0)
        assert model.is_inside(location) is inside

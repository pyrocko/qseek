from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qseek.models.location import Location
from qseek.models.station import Station
from qseek.tracers.fast_marching_3d.velocity_models import (
    NonLinLocHeader,
    NonLinLocVelocityModel,
    VelocityModel3D,
)

KM = 1e3
CENTER = Location(lat=10.0, lon=10.0)


def make_model(grid_spacing: float = 100.0) -> VelocityModel3D:
    model = VelocityModel3D(
        center=CENTER,
        grid_spacing=grid_spacing,
        east_bounds=(-1 * KM, 1 * KM),
        north_bounds=(-2 * KM, 2 * KM),
        depth_bounds=(0 * KM, 1 * KM),
    )
    # Velocity increases with depth
    depths = model.depth_coords
    model.set_velocity_model(
        np.broadcast_to(
            3000.0 + depths[np.newaxis, np.newaxis, :], model.velocity_model.shape
        ).copy()
    )
    return model


def location(east: float, north: float, depth: float) -> Location:
    return Location(
        lat=CENTER.lat,
        lon=CENTER.lon,
        east_shift=east,
        north_shift=north,
        depth=depth,
    )


def test_velocity_model_grid() -> None:
    model = make_model()
    assert model.velocity_model.shape == (21, 41, 11)
    assert model.volume == pytest.approx(2 * KM * 4 * KM * 1 * KM)
    np.testing.assert_allclose(model.east_coords[[0, -1]], (-1 * KM, 1 * KM))

    with pytest.raises(ValueError, match="does not match"):
        model.set_velocity_model(np.zeros((2, 2, 2)))


def test_velocity_model_hash() -> None:
    model = make_model()
    assert model.hash() == make_model().hash()

    other = make_model()
    other.velocity_model[0, 0, 0] = 1.0
    other._hash = None
    assert other.hash() != model.hash()


def test_velocity_model_locations() -> None:
    model = make_model()
    inside = location(310.0, -190.0, 520.0)
    outside = location(5 * KM, 0.0, 0.0)

    assert model.is_inside(inside)
    assert not model.is_inside(outside)
    assert model.get_location_indices(inside) == (13, 18, 5)
    assert model.get_velocity(inside) == pytest.approx(3500.0)
    with pytest.raises(ValueError, match="outside"):
        model.get_location_indices(outside)

    distances = model.get_source_distance_grid(location(0.0, 0.0, 0.0))
    assert distances[10, 20, 0] == pytest.approx(0.0, abs=1e-6)
    assert distances[0, 20, 0] == pytest.approx(1 * KM, rel=1e-6)


def test_velocity_model_crop() -> None:
    model = make_model()
    cropped = model.crop_to_locations(
        [location(-200.0, -300.0, 100.0), location(200.0, 300.0, 300.0)],
        margin=1,
    )
    assert cropped.east_bounds == pytest.approx((-300.0, 300.0))
    assert cropped.north_bounds == pytest.approx((-400.0, 400.0))
    assert cropped.depth_bounds == pytest.approx((0.0, 400.0))
    assert cropped.volume < model.volume
    np.testing.assert_allclose(
        cropped.velocity_model[0, 0], 3000.0 + np.arange(5) * 100
    )


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_velocity_model_resample(method: str) -> None:
    model = make_model()
    assert model.resample(100.0) is model

    resampled = model.resample(50.0, method=method)
    assert resampled.velocity_model.shape == (41, 81, 21)
    assert resampled.velocity_model.min() == pytest.approx(3000.0)
    assert resampled.velocity_model.max() == pytest.approx(4000.0)
    if method != "nearest":
        np.testing.assert_allclose(
            resampled.velocity_model[0, 0], 3000.0 + resampled.depth_coords, rtol=1e-5
        )


def test_velocity_model_export_vtk(tmp_path: Path) -> None:
    model = make_model(grid_spacing=500.0)
    model.export_vtk(tmp_path / "model", reference=location(100.0, 0.0, 0.0))
    assert (tmp_path / "model.vtr").exists()


def write_nll_model(
    path: Path,
    header: str,
    data: np.ndarray | None = None,
) -> Path:
    header_file = path / "model.P.mod.hdr"
    header_file.write_text(header + "\nTRANSFORM  NONE\n")
    if data is not None:
        data.tofile(header_file.with_suffix(".buf"))
    return header_file


def test_nll_header(tmp_path: Path) -> None:
    file = write_nll_model(
        tmp_path, "3 4 5  -1.0 -2.0 -0.5  0.1 0.1 0.1 VELOCITY_METERS DOUBLE"
    )
    # The header's origin is in km relative to an origin it does not hold
    with pytest.raises(ValueError, match="reference_location"):
        NonLinLocHeader.from_header_file(file)

    header = NonLinLocHeader.from_header_file(file, reference_location=CENTER)
    assert (header.nx, header.ny, header.nz) == (3, 4, 5)
    assert header.grid_spacing == pytest.approx(100.0)
    assert header.dtype is float
    assert header.east_bounds == pytest.approx((-100.0, 100.0))
    assert header.north_bounds == pytest.approx((-150.0, 150.0))
    assert header.depth_bounds == pytest.approx((0.0, 400.0))
    assert header.origin.elevation == pytest.approx(CENTER.elevation + 500.0)
    assert header.origin.east_shift == pytest.approx(-1 * KM)
    assert header.origin.north_shift == pytest.approx(-2 * KM)
    assert header.center.east_shift == pytest.approx(-1 * KM + 100.0)
    assert header.center.north_shift == pytest.approx(-2 * KM + 150.0)


def test_nll_header_default_dtype(tmp_path: Path) -> None:
    file = write_nll_model(tmp_path, "3 4 5  0 0 0  0.1 0.1 0.1 SLOW_LEN")
    header = NonLinLocHeader.from_header_file(file, reference_location=CENTER)
    assert header.grid_dtype == "FLOAT"
    assert header.dtype is np.float32


@pytest.mark.parametrize(
    ("header", "match"),
    [
        ("3 4 5  0 0 0  0.1 0.2 0.1 SLOW_LEN FLOAT", "equal spacing"),
        ("3 4 5  0 0 0  0.1 0.1 0.1 SLOW_LEN FLOAT EXTRA", "Invalid"),
        ("3 4 5  0 0 0  0.1 0.1 SLOW_LEN", "Invalid"),
    ],
)
def test_nll_header_invalid(tmp_path: Path, header: str, match: str) -> None:
    file = write_nll_model(tmp_path, header)
    with pytest.raises(ValueError, match=match):
        NonLinLocHeader.from_header_file(file)


def test_nll_header_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Error reading"):
        NonLinLocHeader.from_header_file(tmp_path / "missing.hdr")


@pytest.mark.parametrize(
    ("grid_type", "values", "velocity"),
    [
        ("VELOCITY", 4.0, 4000.0),
        ("VELOCITY_METERS", 4000.0, 4000.0),
        # Slowness times the grid spacing in km
        ("SLOW_LEN", 0.1 / 4.0, 4000.0),
    ],
)
def test_nll_velocity_model_grid_types(
    tmp_path: Path, grid_type: str, values: float, velocity: float
) -> None:
    shape = (5, 6, 7)
    file = write_nll_model(
        tmp_path,
        f"{shape[0]} {shape[1]} {shape[2]}  -0.2 -0.25 0.0  0.1 0.1 0.1"
        f" {grid_type} FLOAT",
        np.full(shape, values, dtype=np.float32),
    )
    model = NonLinLocVelocityModel(
        header_file=file, reference_location=CENTER
    ).prepare()
    assert model._velocity_model.dtype == np.float32
    np.testing.assert_allclose(model._velocity_model, velocity, rtol=1e-5)


def test_nll_velocity_model_errors(tmp_path: Path) -> None:
    file = write_nll_model(tmp_path, "2 2 2  0 0 0  0.1 0.1 0.1 SLOW_LEN FLOAT")
    with pytest.raises(FileNotFoundError):
        NonLinLocVelocityModel(header_file=file, reference_location=CENTER).prepare()

    file = write_nll_model(
        tmp_path,
        "2 2 2  0 0 0  0.1 0.1 0.1 UNKNOWN FLOAT",
        np.ones((2, 2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="grid_type"):
        NonLinLocVelocityModel(header_file=file, reference_location=CENTER).prepare()


def test_nll_velocity_model_get_model(tmp_path: Path, octree) -> None:
    shape = (9, 9, 5)
    file = write_nll_model(
        tmp_path,
        f"{shape[0]} {shape[1]} {shape[2]}  -0.4 -0.4 0.0  0.1 0.1 0.1 VELOCITY FLOAT",
        np.full(shape, 4.0, dtype=np.float32),
    )
    reference = Location(lat=10.0, lon=10.0)
    for grid_spacing, expected in (("input", 100.0), (200.0, 200.0)):
        nll = NonLinLocVelocityModel(
            header_file=file,
            reference_location=reference,
            grid_spacing=grid_spacing,
            interpolation="linear",
        ).prepare()
        model = nll.get_model(octree, None)
        assert model.grid_spacing == expected
        np.testing.assert_allclose(model.velocity_model, 4000.0)

    # Cropped to the stations and the octree
    station = Station(network="XX", station="STA", lat=10.0, lon=10.0)
    model = nll.get_model(octree, [station])
    assert model.grid_spacing == 200.0

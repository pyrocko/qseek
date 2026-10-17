from __future__ import annotations

import random

import numpy as np
import pytest
from pyrocko import orthodrome

from qseek.models import Location

KM = 1e3


def test_location() -> None:
    loc = Location(lat=11.0, lon=23.55)
    loc_other = Location(lat=13.123, lon=21.12)

    distance = loc.surface_distance_to(loc_other)
    assert distance == pytest.approx(
        orthodrome.distance_accurate50m(11.0, 23.55, 13.123, 21.12)
    )
    assert loc_other.surface_distance_to(loc) == pytest.approx(distance)


def test_distance_same_origin():
    loc = Location(lat=11.0, lon=23.55)

    rng = random.Random(0)
    perturb_attributes = {"north_shift", "east_shift", "elevation", "depth"}
    for _ in range(100):
        distance = rng.uniform(-10 * KM, 10 * KM)
        for attr in perturb_attributes:
            loc_other = loc.model_copy()
            loc_other._cached_lat_lon = None
            setattr(loc_other, attr, distance)
            assert loc.distance_to(loc_other) == abs(distance)

            loc_shifted = loc_other.shifted_origin()
            np.testing.assert_approx_equal(
                loc.distance_to(loc_shifted),
                abs(distance),
                significant=2,
            )


def test_location_offset():
    loc = Location(lat=11.0, lon=23.55)
    loc_other = Location(
        lat=11.0,
        lon=23.55,
        north_shift=100.0,
        east_shift=100.0,
        depth=100.0,
    )

    offset = loc_other.offset_from(loc)
    assert offset == (100.0, 100.0, 100.0)

    loc_other = Location(
        lat=11.0,
        lon=23.55,
        north_shift=100.0,
        east_shift=100.0,
        elevation=100.0,
    )
    offset = loc_other.offset_from(loc)
    assert offset == (100.0, 100.0, -100.0)

    loc_other = Location(
        lat=11.0,
        lon=23.55,
        north_shift=100.0,
        east_shift=100.0,
        elevation=100.0,
        depth=10.0,
    )
    offset = loc_other.offset_from(loc)
    assert offset == (100.0, 100.0, -90.0)

    loc2 = Location(
        lat=11.0,
        lon=23.55,
        elevation=20.0,
        depth=20.0,
    )
    loc_other = Location(
        lat=11.0,
        lon=23.55,
        north_shift=100.0,
        east_shift=100.0,
        elevation=100.0,
        depth=10.0,
    )
    offset = loc_other.offset_from(loc2)
    assert offset == (100.0, 100.0, -90.0)

    loc_other = loc_other.shifted_origin()
    offset = loc_other.offset_from(loc2)
    np.testing.assert_almost_equal(offset, (100.0, 100.0, -90.0), decimal=0)


def test_absolute_locations():
    degree_in_m = 111319.54315315

    loc = Location(lat=0, lon=0)
    loc_other = Location(lat=0, lon=1)
    np.testing.assert_almost_equal(loc_other.offset_from(loc), (degree_in_m, 0, 0))

    loc = Location(lat=0, lon=0, depth=0)
    loc_other = Location(lat=0, lon=1, depth=100)
    np.testing.assert_almost_equal(loc_other.offset_from(loc), (degree_in_m, 0, 100))

    loc = Location(lat=0, lon=0)
    loc_other = Location(lat=0, lon=1, east_shift=5 * KM)
    np.testing.assert_almost_equal(
        loc_other.offset_from(loc), (degree_in_m + 5 * KM, 0, 0)
    )

    loc = Location(lat=0, lon=0, east_shift=1 * KM)
    loc_other = Location(lat=0, lon=1, east_shift=5 * KM)
    np.testing.assert_almost_equal(
        loc_other.offset_from(loc), (degree_in_m + 4 * KM, 0, 0)
    )

    loc = Location(lat=0, lon=0, east_shift=1 * KM)
    loc_other = Location(lat=0, lon=1, east_shift=5 * KM, elevation=1 * KM)
    np.testing.assert_almost_equal(
        loc_other.offset_from(loc), (degree_in_m + 4 * KM, 0, -1 * KM)
    )

    loc = Location(lat=0, lon=0, east_shift=1 * KM, depth=0.5 * KM)
    loc_other = Location(lat=0, lon=1, east_shift=5 * KM, elevation=1 * KM)
    np.testing.assert_almost_equal(
        loc_other.offset_from(loc), (degree_in_m + 4 * KM, 0, -1.5 * KM)
    )


def test_location_offset_frame():
    # The offset from a reference is a vector in the reference's frame
    from pyrocko import orthodrome as od

    reference = Location(lat=42.8, lon=13.2, east_shift=300.0, north_shift=-200.0)
    station = Location(lat=43.3, lon=13.9)
    east, north, _ = station.offset_from(reference)
    ref_north, ref_east = od.latlon_to_ne_numpy(
        reference.lat, reference.lon, station.lat, station.lon
    )
    np.testing.assert_allclose(east, ref_east[0] - reference.east_shift)
    np.testing.assert_allclose(north, ref_north[0] - reference.north_shift)

    # Reversing the bearing at the station rotates it by the meridian convergence
    back_north, back_east = od.latlon_to_ne_numpy(
        station.lat, station.lon, reference.lat, reference.lon
    )
    rotated = np.hypot(
        -back_east[0] - reference.east_shift - east,
        -back_north[0] - reference.north_shift - north,
    )
    assert rotated > 100.0


def test_location_cache_follows_coordinates():
    location = Location(lat=42.0, lon=13.0, east_shift=1000.0)
    lat_lon = location.effective_lat_lon
    origin = location.origin()

    moved = location.shift(east=5 * KM, north=0.0, elevation=200.0)
    expected = Location(lat=42.0, lon=13.0, east_shift=6 * KM, elevation=200.0)
    assert moved.effective_lat_lon == expected.effective_lat_lon
    assert moved.effective_lat_lon != lat_lon
    assert moved.origin().elevation == 200.0
    assert location.origin() is origin

    copied = location.model_copy(update={"north_shift": 2 * KM})
    assert (
        copied.effective_lat_lon
        == (
            Location(lat=42.0, lon=13.0, east_shift=1000.0, north_shift=2 * KM)
        ).effective_lat_lon
    )

from __future__ import annotations

import random

import numpy as np

from qseek.models import Location

KM = 1e3


def test_location() -> None:
    loc = Location(lat=11.0, lon=23.55)
    loc_other = Location(lat=13.123, lon=21.12)

    loc.surface_distance_to(loc_other)


def test_distance_same_origin():
    loc = Location(lat=11.0, lon=23.55)

    perturb_attributes = {"north_shift", "east_shift", "elevation", "depth"}
    for _ in range(100):
        distance = random.uniform(-10 * KM, 10 * KM)
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

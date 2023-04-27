from lassie.models import Location


def test_location() -> None:
    loc = Location(lat=11.0, lon=23.55)
    loc_other = Location(lat=13.123, lon=21.12)

    loc.surface_distance_to(loc_other)

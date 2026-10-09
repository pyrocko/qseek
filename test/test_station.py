from __future__ import annotations

from qseek.models.station import Station, StationInventory


def station(name: str, lat: float = 10.0) -> Station:
    return Station(network="XX", station=name, location="", lat=lat, lon=10.0)


def test_station_inventory_hash() -> None:
    """The hash is stable and follows the stations."""
    inventory = StationInventory.model_construct(stations=[station("STA")])
    same = StationInventory.model_construct(stations=[station("STA")])
    other = StationInventory.model_construct(stations=[station("STB")])
    moved = StationInventory.model_construct(stations=[station("STA", lat=11.0)])

    # Hashes of generators differ while the generators are alive
    hashes = [hash(inventory) for _ in range(5)]
    assert len(set(hashes)) == 1
    assert inventory in {inventory}
    assert {inventory: 1}[inventory] == 1

    assert hash(inventory) == hash(same)
    assert hash(inventory) != hash(other)
    assert hash(inventory) != hash(moved)

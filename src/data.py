"""Load and provide access to project data files."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Station:
    name: str
    lat: float
    lon: float
    speed_kmh: float


def load_stations() -> list[Station]:
    with open(DATA_DIR / "stations.json") as f:
        raw = json.load(f)
    return [Station(**s) for s in raw]


def load_stations_raw() -> list[dict]:
    """Raw dicts for pydeck layers."""
    with open(DATA_DIR / "stations.json") as f:
        return json.load(f)


def load_passages() -> dict:
    with open(DATA_DIR / "passages.json") as f:
        return json.load(f)


def load_zones_geojson() -> dict:
    with open(DATA_DIR / "neva_zone.geojson") as f:
        return json.load(f)


def load_water_polygon():
    """Union of all water zone polygons as a single Shapely geometry."""
    geojson = load_zones_geojson()
    polys = [shape(f["geometry"]) for f in geojson["features"]]
    return unary_union(polys)


def load_zone_polygons() -> list[tuple[str, any]]:
    """List of (zone_name, shapely_polygon) pairs."""
    geojson = load_zones_geojson()
    return [(f["properties"]["zone"], shape(f["geometry"])) for f in geojson["features"]]


def classify_cells_by_zone(
    lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Assign each grid cell to 'N' (north) or 'S' (south) zone.

    Uses the zone polygons; cells that fall in a 'north' polygon get 'N',
    everything else gets 'S'.
    """
    from shapely.geometry import Point
    from shapely.prepared import prep as sprep

    zone_polys = load_zone_polygons()
    north_polys = [p for name, p in zone_polys if name == "north"]
    north = unary_union(north_polys)
    north_prep = sprep(north)

    result = np.empty(len(lats), dtype="U1")
    for i in range(len(lats)):
        result[i] = "N" if north_prep.contains(Point(lons[i], lats[i])) else "S"
    return result


def get_passage_coords() -> list[tuple[float, float]]:
    """Return passage coordinates as [(lat, lon), ...]."""
    passages = load_passages()
    return [(v["lat"], v["lon"]) for v in passages.values()]

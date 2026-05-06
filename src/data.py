"""Load and provide access to project data files."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Station:
    name: str
    lat: float
    lon: float
    speed_kmh: float
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = self.name


def load_stations() -> list[Station]:
    with open(DATA_DIR / "stations.json", encoding="utf-8") as f:
        raw = json.load(f)
    stations = [Station(**s) for s in raw]
    ids = [s.id for s in stations]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate station ids in stations.json: {ids}")
    return stations


def load_stations_raw() -> list[dict]:
    """Raw dicts for pydeck layers."""
    with open(DATA_DIR / "stations.json", encoding="utf-8") as f:
        return json.load(f)


def load_passages() -> dict:
    with open(DATA_DIR / "passages.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_risk_scenarios() -> dict:
    with open(DATA_DIR / "risk_scenarios.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_zones_geojson() -> dict:
    with open(DATA_DIR / "neva_zone.geojson") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_shoreline_geojson() -> dict:
    with open(DATA_DIR / "shoreline.geojson", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_shoreline():
    """Load coastline used for shore-distance risk components."""
    geojson = load_shoreline_geojson()
    lines = [shape(f["geometry"]) for f in geojson["features"]]
    return unary_union(lines)


@lru_cache(maxsize=1)
def load_water_polygon():
    """Union of all water zone polygons as a single Shapely geometry."""
    geojson = load_zones_geojson()
    polys = [shape(f["geometry"]) for f in geojson["features"]]
    return unary_union(polys)


def load_zone_polygons() -> list[tuple[str, any]]:
    """List of (zone_name, shapely_polygon) pairs."""
    geojson = load_zones_geojson()
    return [
        (f["properties"]["zone"], shape(f["geometry"])) for f in geojson["features"]
    ]


@lru_cache(maxsize=1)
def _north_zone_union():
    return unary_union([p for name, p in load_zone_polygons() if name == "north"])


def classify_cells_by_zone(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Assign each grid cell to 'N' (north) or 'S' (south) zone."""
    north = _north_zone_union()
    in_north = contains_xy(north, np.asarray(lons), np.asarray(lats))
    return np.where(in_north, "N", "S").astype("U1")


def get_passage_coords() -> list[tuple[float, float]]:
    """Return passage coordinates as [(lat, lon), ...]."""
    passages = load_passages()
    return [(v["lat"], v["lon"]) for v in passages.values()]

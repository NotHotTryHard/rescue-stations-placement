"""Load and provide access to project data files."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, MultiLineString, Point, box, shape
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


# Bounding box around Kronshtadt island (incl. КЗС causeway segments alongside it).
# Used to extract the island outline from zone outer rings.
KRONSHTADT_BBOX = (29.62, 59.97, 29.83, 60.04)  # (lon_min, lat_min, lon_max, lat_max)

# Minimum length (in degrees) for a Kronshtadt outline segment to be kept.
# Drops tiny artifact runs (~21 pts near С-2 entry).
_KRONSHTADT_MIN_PART_LEN_DEG = 0.005

# Tolerance for detecting whether a fragment touches an "ear tip" (endpoint of
# the original outer-ring run inside the bbox). ~10m at this latitude.
_KRONSHTADT_EAR_TOL_DEG = 1e-4


@lru_cache(maxsize=1)
def load_kronshtadt_cuts():
    """User-provided line segments that trim КЗС "ears" from the Kronshtadt outline."""
    with open(DATA_DIR / "kronshtadt_cuts.geojson", encoding="utf-8") as f:
        return json.load(f)


def _extract_runs_in_bbox(bbox) -> list[LineString]:
    runs: list[LineString] = []
    for _zone_name, polygon in load_zone_polygons():
        coords = list(polygon.exterior.coords)
        cur: list[tuple[float, float]] = []
        for x, y in coords:
            if bbox.covers(Point(x, y)):
                cur.append((x, y))
            else:
                if len(cur) >= 2:
                    runs.append(LineString(cur))
                cur = []
        if len(cur) >= 2:
            runs.append(LineString(cur))
    return runs


@lru_cache(maxsize=1)
def load_kronshtadt_outline():
    """Outline of Kronshtadt as a MultiLineString.

    Pipeline:
    1. Take north + south zone outer rings restricted to KRONSHTADT_BBOX.
    2. Split each run at intersections with cuts from `kronshtadt_cuts.geojson`.
    3. Drop fragments that touch an original run endpoint — these are the КЗС
       causeway "ears" approaching С-1 / С-2.
    4. Drop fragments whose both endpoints lie on the SAME cut — these are
       inlets (ports) where the contour wraps in and out across one cut line.
    5. Drop fragments shorter than `_KRONSHTADT_MIN_PART_LEN_DEG`.
    """
    from shapely.ops import split, unary_union

    bbox = box(*KRONSHTADT_BBOX)
    raw_runs = _extract_runs_in_bbox(bbox)
    if not raw_runs:
        raise RuntimeError("Kronshtadt outline extraction returned no segments")

    ear_tips = []
    for run in raw_runs:
        ear_tips.append(Point(run.coords[0]))
        ear_tips.append(Point(run.coords[-1]))

    cuts_geojson = load_kronshtadt_cuts()
    cut_lines = [shape(f["geometry"]) for f in cuts_geojson["features"]]
    cuts_union = unary_union(cut_lines) if cut_lines else None

    def closest_cut_idx(pt: Point) -> int | None:
        if not cut_lines:
            return None
        dists = [pt.distance(c) for c in cut_lines]
        return int(min(range(len(cut_lines)), key=lambda i: dists[i]))

    kept: list[LineString] = []
    for run in raw_runs:
        pieces = [run]
        if cuts_union is not None and run.intersects(cuts_union):
            split_result = split(run, cuts_union)
            pieces = [g for g in split_result.geoms if isinstance(g, LineString)]
        for piece in pieces:
            if piece.length < _KRONSHTADT_MIN_PART_LEN_DEG:
                continue
            if any(piece.distance(t) < _KRONSHTADT_EAR_TOL_DEG for t in ear_tips):
                continue
            # Detect inlet: both ends lie on the same cut LineString (within tol).
            if cut_lines:
                a = Point(piece.coords[0])
                b = Point(piece.coords[-1])
                ia, ib = closest_cut_idx(a), closest_cut_idx(b)
                if ia is not None and ia == ib:
                    da = a.distance(cut_lines[ia])
                    db = b.distance(cut_lines[ib])
                    if da < _KRONSHTADT_EAR_TOL_DEG and db < _KRONSHTADT_EAR_TOL_DEG:
                        continue
            kept.append(piece)

    if not kept:
        raise RuntimeError("Kronshtadt outline: nothing left after cuts")
    return MultiLineString(kept)


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

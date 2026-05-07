"""Sample candidate placements along the mainland shore and the Kronshtadt outline.

Two physical sources:
- Mainland coast — `load_shoreline()` (curated north + south mainland LineStrings).
- Kronshtadt outline — `load_kronshtadt_outline()` (extracted from zone outer rings
  inside the Kronshtadt bbox; includes adjacent КЗС causeway segments).

`sample_shore_candidates` returns the union of both. Other water-boundary points
(small islands, the dam outside Kronshtadt) are intentionally excluded — placing a
rescue station there is not physically meaningful.
"""

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform

from ..data import load_kronshtadt_outline, load_shoreline
from ..grid import METERS_PER_DEG_LAT, METERS_PER_DEG_LON
from .placement import PlacementSet, attach_travel_times


def _to_meters(geom):
    return transform(
        lambda lon, lat, z=None: (lon * METERS_PER_DEG_LON, lat * METERS_PER_DEG_LAT),
        geom,
    )


def _from_meters(x: float, y: float) -> tuple[float, float]:
    return y / METERS_PER_DEG_LAT, x / METERS_PER_DEG_LON


def _iter_linestrings(geom) -> list[LineString]:
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        out = []
        for sub in geom.geoms:
            out.extend(_iter_linestrings(sub))
        return out
    raise TypeError(f"unsupported boundary geometry: {type(geom).__name__}")


def _sample_along(
    geom,
    step_m: float,
    min_segment_m: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk every linestring in `geom` (lon/lat) and emit a point every step_m meters."""
    if step_m <= 0:
        raise ValueError("step_m must be positive")

    geom_m = _to_meters(geom)
    lats: list[float] = []
    lons: list[float] = []
    for line in _iter_linestrings(geom_m):
        L = line.length
        if L < min_segment_m:
            continue
        n_steps = max(1, int(np.floor(L / step_m)))
        for k in range(n_steps):
            pt = line.interpolate(k * step_m)
            la, lo = _from_meters(pt.x, pt.y)
            lats.append(la)
            lons.append(lo)
    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


def sample_mainland_points(step_m: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample the mainland (`shoreline.geojson`) at constant arclength."""
    return _sample_along(load_shoreline(), step_m=step_m)


def sample_kronshtadt_points(step_m: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample the Kronshtadt outline (extracted from zone outer rings)."""
    return _sample_along(load_kronshtadt_outline(), step_m=step_m)


def _build(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    speed_kmh: float,
    label_prefix: str,
    graph: csr_matrix,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    exclude_grid_indices: Sequence[int],
) -> PlacementSet:
    speed = np.full(len(lat), float(speed_kmh), dtype=np.float64)
    labels = [f"{label_prefix}_{i:04d}" for i in range(len(lat))]
    placements = attach_travel_times(
        lat=lat, lon=lon, speed_kmh=speed, labels=labels,
        graph=graph, grid_lats=grid_lats, grid_lons=grid_lons,
    )
    if len(exclude_grid_indices):
        excl = {int(i) for i in exclude_grid_indices}
        keep = np.array([int(i) not in excl for i in placements.grid_index], dtype=bool)
        placements = _select(placements, keep)
    # Dedupe candidates that snapped to the same grid cell — keep first
    _, first_idx = np.unique(placements.grid_index, return_index=True)
    first_idx = np.sort(first_idx)
    if len(first_idx) != placements.K:
        placements = _select_by_index(placements, first_idx)
    return placements


def _select(p: PlacementSet, mask: np.ndarray) -> PlacementSet:
    return PlacementSet(
        lat=p.lat[mask], lon=p.lon[mask], speed_kmh=p.speed_kmh[mask],
        grid_index=p.grid_index[mask], travel_times=p.travel_times[mask],
        labels=[lbl for lbl, k in zip(p.labels, mask) if k],
    )


def _select_by_index(p: PlacementSet, idx: np.ndarray) -> PlacementSet:
    return PlacementSet(
        lat=p.lat[idx], lon=p.lon[idx], speed_kmh=p.speed_kmh[idx],
        grid_index=p.grid_index[idx], travel_times=p.travel_times[idx],
        labels=[p.labels[int(i)] for i in idx],
    )


def _concat(a: PlacementSet, b: PlacementSet) -> PlacementSet:
    return PlacementSet(
        lat=np.concatenate([a.lat, b.lat]),
        lon=np.concatenate([a.lon, b.lon]),
        speed_kmh=np.concatenate([a.speed_kmh, b.speed_kmh]),
        grid_index=np.concatenate([a.grid_index, b.grid_index]),
        travel_times=np.concatenate([a.travel_times, b.travel_times], axis=0),
        labels=list(a.labels) + list(b.labels),
    )


def sample_mainland_candidates(
    *,
    step_m: float = 300.0,
    speed_kmh: float = 40.0,
    graph: csr_matrix,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    exclude_grid_indices: Sequence[int] = (),
) -> PlacementSet:
    lat, lon = sample_mainland_points(step_m=step_m)
    return _build(
        lat=lat, lon=lon, speed_kmh=speed_kmh, label_prefix="main",
        graph=graph, grid_lats=grid_lats, grid_lons=grid_lons,
        exclude_grid_indices=exclude_grid_indices,
    )


def sample_kronshtadt_candidates(
    *,
    step_m: float = 300.0,
    speed_kmh: float = 40.0,
    graph: csr_matrix,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    exclude_grid_indices: Sequence[int] = (),
) -> PlacementSet:
    lat, lon = sample_kronshtadt_points(step_m=step_m)
    return _build(
        lat=lat, lon=lon, speed_kmh=speed_kmh, label_prefix="kron",
        graph=graph, grid_lats=grid_lats, grid_lons=grid_lons,
        exclude_grid_indices=exclude_grid_indices,
    )


def sample_shore_candidates(
    *,
    step_m: float = 300.0,
    speed_kmh: float = 40.0,
    graph: csr_matrix,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    exclude_grid_indices: Sequence[int] = (),
) -> PlacementSet:
    """Mainland coast + Kronshtadt outline, both at the same `step_m`."""
    mainland = sample_mainland_candidates(
        step_m=step_m, speed_kmh=speed_kmh,
        graph=graph, grid_lats=grid_lats, grid_lons=grid_lons,
        exclude_grid_indices=exclude_grid_indices,
    )
    kron = sample_kronshtadt_candidates(
        step_m=step_m, speed_kmh=speed_kmh,
        graph=graph, grid_lats=grid_lats, grid_lons=grid_lons,
        exclude_grid_indices=tuple(int(i) for i in exclude_grid_indices)
        + tuple(int(i) for i in mainland.grid_index),
    )
    return _concat(mainland, kron)

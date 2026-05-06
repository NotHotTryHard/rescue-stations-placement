"""Sample candidate placements along the shore (boundary of water polygons)."""

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform

from ..data import load_water_polygon
from ..grid import METERS_PER_DEG_LAT, METERS_PER_DEG_LON
from .placement import PlacementSet, attach_travel_times


def _to_meters(geom):
    return transform(lambda lon, lat, z=None: (lon * METERS_PER_DEG_LON, lat * METERS_PER_DEG_LAT), geom)


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


def sample_shore_points(step_m: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """Walk along the water boundary in local meters and sample every `step_m`.

    Returns (lats, lons). Includes outer shores AND interior holes (Кронштадт).
    """
    if step_m <= 0:
        raise ValueError("step_m must be positive")

    water = load_water_polygon()
    boundary_m = _to_meters(water.boundary)

    lats: list[float] = []
    lons: list[float] = []
    for line in _iter_linestrings(boundary_m):
        L = line.length
        if L <= 0:
            continue
        n_steps = max(1, int(np.floor(L / step_m)))
        for k in range(n_steps):
            pt = line.interpolate(k * step_m)
            la, lo = _from_meters(pt.x, pt.y)
            lats.append(la)
            lons.append(lo)

    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


def sample_shore_candidates(
    *,
    step_m: float = 300.0,
    speed_kmh: float = 40.0,
    graph: csr_matrix,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    exclude_grid_indices: Sequence[int] = (),
) -> PlacementSet:
    """Sample shore candidates and attach precomputed travel times."""
    lat, lon = sample_shore_points(step_m=step_m)
    speed = np.full(len(lat), float(speed_kmh), dtype=np.float64)
    labels = [f"shore_{i:04d}" for i in range(len(lat))]

    placements = attach_travel_times(
        lat=lat, lon=lon, speed_kmh=speed, labels=labels,
        graph=graph, grid_lats=grid_lats, grid_lons=grid_lons,
    )

    # Drop candidates that snap onto a cell already occupied by an existing station
    if len(exclude_grid_indices):
        excl = set(int(i) for i in exclude_grid_indices)
        keep = np.array([int(i) not in excl for i in placements.grid_index], dtype=bool)
        if not keep.all():
            placements = PlacementSet(
                lat=placements.lat[keep],
                lon=placements.lon[keep],
                speed_kmh=placements.speed_kmh[keep],
                grid_index=placements.grid_index[keep],
                travel_times=placements.travel_times[keep],
                labels=[lbl for lbl, k in zip(placements.labels, keep) if k],
            )

    # Deduplicate candidates that snapped to the same grid cell — keep first
    _, first_idx = np.unique(placements.grid_index, return_index=True)
    first_idx = np.sort(first_idx)
    if len(first_idx) != placements.K:
        placements = PlacementSet(
            lat=placements.lat[first_idx],
            lon=placements.lon[first_idx],
            speed_kmh=placements.speed_kmh[first_idx],
            grid_index=placements.grid_index[first_idx],
            travel_times=placements.travel_times[first_idx],
            labels=[placements.labels[i] for i in first_idx],
        )

    return placements

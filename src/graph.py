"""Build a navigation graph from the water grid."""

import numpy as np
from scipy.sparse import csr_matrix

from .grid import METERS_PER_DEG_LAT, METERS_PER_DEG_LON


def build_graph(
    lats: np.ndarray,
    lons: np.ndarray,
    dlat: float,
    dlon: float,
    cell_zones: np.ndarray | None = None,
    passage_coords: list[tuple[float, float]] | None = None,
    passage_radius_m: float = 1000.0,
) -> csr_matrix:
    """Build a sparse adjacency matrix connecting neighboring water cells.

    Uses 8-connectivity (horizontal, vertical, diagonal neighbors).
    Edge weights are distances in meters.

    When ``cell_zones`` is provided, cross-zone edges are only allowed
    near passage coordinates (within ``passage_radius_m``).  This
    prevents paths from crossing the dam except through С-1 / С-2.

    Parameters
    ----------
    lats, lons : ndarray of shape (N,)
        Coordinates of water grid cells.
    dlat, dlon : float
        Grid step sizes in degrees.
    cell_zones : ndarray of shape (N,), optional
        Zone label for each cell (e.g. ``"N"`` / ``"S"``).
    passage_coords : list of (lat, lon), optional
        Coordinates of allowed cross-zone passages.
    passage_radius_m : float
        Max distance from a passage for a cross-zone edge to be kept.

    Returns
    -------
    graph : csr_matrix of shape (N, N)
        Sparse adjacency matrix with distances in meters as weights.
    """
    n = len(lats)

    # Build a spatial index: map (row, col) grid indices to cell index
    lat_min = lats.min()
    lon_min = lons.min()
    rows = np.rint((lats - lat_min) / dlat).astype(np.int32)
    cols = np.rint((lons - lon_min) / dlon).astype(np.int32)

    cell_map = {}
    for i in range(n):
        cell_map[(rows[i], cols[i])] = i

    # 8-connected neighbors: (drow, dcol, distance_factor)
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.4142135623730951),
        (-1, 1, 1.4142135623730951),
        (1, -1, 1.4142135623730951),
        (1, 1, 1.4142135623730951),
    ]

    cell_lat_m = dlat * METERS_PER_DEG_LAT
    cell_lon_m = dlon * METERS_PER_DEG_LON
    cell_m = (cell_lat_m + cell_lon_m) / 2.0

    # Pre-check: is cross-zone filtering needed?
    filter_zones = cell_zones is not None and passage_coords is not None

    src_list = []
    dst_list = []
    weight_list = []

    for i in range(n):
        r, c = rows[i], cols[i]
        for dr, dc, dist_factor in neighbors:
            j = cell_map.get((r + dr, c + dc))
            if j is None:
                continue

            # Cross-zone check: block edges between zones unless near a passage
            if filter_zones and cell_zones[i] != cell_zones[j]:
                mid_lat = (lats[i] + lats[j]) / 2.0
                mid_lon = (lons[i] + lons[j]) / 2.0
                if not _near_any_passage(
                    mid_lat, mid_lon, passage_coords, passage_radius_m
                ):
                    continue

            src_list.append(i)
            dst_list.append(j)
            weight_list.append(cell_m * dist_factor)

    return csr_matrix(
        (np.array(weight_list, dtype=np.float32), (src_list, dst_list)),
        shape=(n, n),
    )


def _near_any_passage(
    lat: float,
    lon: float,
    passages: list[tuple[float, float]],
    radius_m: float,
) -> bool:
    """Check if a point is within radius_m of any passage."""
    for p_lat, p_lon in passages:
        dy = (lat - p_lat) * METERS_PER_DEG_LAT
        dx = (lon - p_lon) * METERS_PER_DEG_LON
        if dy * dy + dx * dx <= radius_m * radius_m:
            return True
    return False

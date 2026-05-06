"""Placement set: candidates / existing stations + their precomputed travel times."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from ..data import Station
from ..grid import snap_to_grid
from ..routing import compute_travel_times


@dataclass
class PlacementSet:
    """K placements with arrival-time vectors over the same N-cell grid."""

    lat: np.ndarray         # (K,)
    lon: np.ndarray         # (K,)
    speed_kmh: np.ndarray   # (K,)
    grid_index: np.ndarray  # (K,) snap into the grid
    travel_times: np.ndarray  # (K, N) minutes
    labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        K = len(self.lat)
        if not (len(self.lon) == len(self.speed_kmh) == len(self.grid_index) == K):
            raise ValueError("lat/lon/speed/grid_index lengths mismatch")
        if self.travel_times.shape[0] != K:
            raise ValueError("travel_times.shape[0] must equal K")
        if not self.labels:
            self.labels = [f"#{i}" for i in range(K)]
        elif len(self.labels) != K:
            raise ValueError("labels length mismatch")

    @property
    def K(self) -> int:
        return len(self.lat)

    @property
    def N(self) -> int:
        return self.travel_times.shape[1]


def attach_travel_times(
    *,
    lat: Sequence[float],
    lon: Sequence[float],
    speed_kmh: Sequence[float],
    labels: Sequence[str],
    graph: csr_matrix,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> PlacementSet:
    """Snap (lat, lon) to grid, run batched Dijkstra, return populated PlacementSet."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    speed = np.asarray(speed_kmh, dtype=np.float64)
    labels = list(labels)
    if not (len(lat) == len(lon) == len(speed) == len(labels)):
        raise ValueError("lat/lon/speed/labels lengths mismatch")
    if len(lat) == 0:
        n = len(grid_lats)
        return PlacementSet(
            lat=lat, lon=lon, speed_kmh=speed,
            grid_index=np.empty(0, dtype=np.int64),
            travel_times=np.empty((0, n), dtype=np.float64),
            labels=labels,
        )

    grid_index = np.array(
        [snap_to_grid(la, lo, grid_lats, grid_lons) for la, lo in zip(lat, lon)],
        dtype=np.int64,
    )

    # Detect duplicate snap targets (different inputs collapse to one cell)
    _, first_idx, counts = np.unique(grid_index, return_index=True, return_counts=True)
    if (counts > 1).any():
        dup_cells = grid_index[np.sort(first_idx[counts > 1])]
        print(
            f"[placement] warning: {int((counts > 1).sum())} grid cells receive "
            f"multiple placements (e.g. cells {dup_cells[:5].tolist()}); "
            "their travel-time rows will be identical."
        )

    times = compute_travel_times(graph, grid_index.tolist(), speed.tolist())
    return PlacementSet(
        lat=lat, lon=lon, speed_kmh=speed,
        grid_index=grid_index, travel_times=times, labels=labels,
    )


def from_stations(
    stations: Sequence[Station],
    *,
    graph: csr_matrix,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> PlacementSet:
    """Build a PlacementSet from project Station records."""
    return attach_travel_times(
        lat=[s.lat for s in stations],
        lon=[s.lon for s in stations],
        speed_kmh=[s.speed_kmh for s in stations],
        labels=[s.id for s in stations],
        graph=graph,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
    )

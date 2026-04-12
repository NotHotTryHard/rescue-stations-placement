"""Shortest path computation using scipy's C-implemented Dijkstra."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def compute_travel_times(
    graph: csr_matrix,
    source_indices: list[int],
    speeds_kmh: list[float],
) -> np.ndarray:
    """Compute travel time from each source to every grid cell.

    Parameters
    ----------
    graph : csr_matrix
        Adjacency matrix with distances in meters.
    source_indices : list[int]
        Grid cell indices of source stations.
    speeds_kmh : list[float]
        Speed in km/h for each source station.

    Returns
    -------
    times : ndarray of shape (len(source_indices), N)
        Travel time in minutes from each source to each cell.
        np.inf for unreachable cells.
    """
    n_cells = graph.shape[0]
    n_sources = len(source_indices)
    times = np.full((n_sources, n_cells), np.inf, dtype=np.float64)

    for i, (src_idx, speed) in enumerate(zip(source_indices, speeds_kmh)):
        # Dijkstra returns distances in the same units as graph weights (meters)
        dist = dijkstra(graph, indices=src_idx, directed=False)
        # Convert meters to minutes: dist_m / (speed_km/h * 1000 / 60)
        speed_m_per_min = speed * 1000.0 / 60.0
        times[i] = dist / speed_m_per_min

    return times

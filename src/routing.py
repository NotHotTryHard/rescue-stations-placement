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
    sources = np.asarray(source_indices, dtype=np.int64)
    speeds = np.asarray(speeds_kmh, dtype=np.float64)
    # One batched Dijkstra call: returns (len(sources), n_cells) in meters
    dist = dijkstra(graph, indices=sources, directed=False)
    speed_m_per_min = speeds * (1000.0 / 60.0)
    return dist / speed_m_per_min[:, None]

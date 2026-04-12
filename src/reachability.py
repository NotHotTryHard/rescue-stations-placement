"""Reachability analysis: minimum time to reach any point from the nearest station."""

import numpy as np


def compute_reachability(travel_times: np.ndarray) -> np.ndarray:
    """Compute minimum travel time to each cell from any station.

    Parameters
    ----------
    travel_times : ndarray of shape (n_stations, n_cells)
        Travel time from each station to each cell.

    Returns
    -------
    min_times : ndarray of shape (n_cells,)
        Minimum time in minutes to reach each cell from the nearest station.
    """
    return np.min(travel_times, axis=0)


def nearest_station_ids(travel_times: np.ndarray) -> np.ndarray:
    """For each cell, which station reaches it fastest.

    Returns
    -------
    station_ids : ndarray of shape (n_cells,)
        Index of the station with the shortest travel time to each cell.
    """
    return np.argmin(travel_times, axis=0)

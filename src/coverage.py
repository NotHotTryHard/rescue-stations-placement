"""Coverage analysis: aggregate reachability into actionable metrics."""

import numpy as np


def coverage_curve(
    min_times: np.ndarray,
    max_time: float = 60.0,
    step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative coverage: % of area reachable within T minutes.

    Returns
    -------
    thresholds : ndarray — time values in minutes
    coverage_pct : ndarray — % of reachable cells covered at each threshold
    """
    reachable = min_times[np.isfinite(min_times)]
    thresholds = np.arange(0, max_time + step, step)
    coverage_pct = np.array(
        [(reachable <= t).sum() / len(reachable) * 100 for t in thresholds]
    )
    return thresholds, coverage_pct


def coverage_at_thresholds(
    min_times: np.ndarray,
    thresholds: list[float] = [5, 10, 15, 20, 25, 30],
) -> list[tuple[float, float]]:
    """Coverage % at specific time thresholds.

    Returns list of (threshold_min, coverage_pct).
    """
    reachable = min_times[np.isfinite(min_times)]
    n = len(reachable)
    return [(t, (reachable <= t).sum() / n * 100) for t in thresholds]


def station_zones(
    travel_times: np.ndarray,
    min_times: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Responsibility zones: which station is nearest to each cell.

    Returns
    -------
    assignments : ndarray of shape (n_cells,)
        Station index for each cell (-1 if unreachable).
    zone_sizes : list of (station_idx, cell_count)
        Number of cells assigned to each station, sorted descending.
    """
    assignments = np.argmin(travel_times, axis=0)
    assignments[~np.isfinite(min_times)] = -1

    n_stations = travel_times.shape[0]
    zone_sizes = []
    for s in range(n_stations):
        count = (assignments == s).sum()
        zone_sizes.append((s, int(count)))
    zone_sizes.sort(key=lambda x: x[1], reverse=True)

    return assignments, zone_sizes


def blind_spots(
    min_times: np.ndarray,
    threshold_min: float = 20.0,
) -> np.ndarray:
    """Indices of cells with response time above threshold (or unreachable)."""
    return np.where((min_times > threshold_min) | ~np.isfinite(min_times))[0]

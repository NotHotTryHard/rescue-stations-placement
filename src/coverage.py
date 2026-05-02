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


def weighted_coverage_curve(
    min_times: np.ndarray,
    weights: np.ndarray,
    max_time: float = 60.0,
    step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative coverage of incident probability within T minutes."""
    times = np.asarray(min_times, dtype=np.float64)
    w = _normalized_weights(weights, len(times))
    thresholds = np.arange(0, max_time + step, step)
    coverage_pct = np.array(
        [
            w[np.isfinite(times) & (times <= t)].sum() * 100
            for t in thresholds
        ]
    )
    return thresholds, coverage_pct


def weighted_coverage_at_thresholds(
    min_times: np.ndarray,
    weights: np.ndarray,
    thresholds: list[float] = [5, 10, 15, 20, 25, 30],
) -> list[tuple[float, float]]:
    """Incident probability covered within specific time thresholds."""
    times = np.asarray(min_times, dtype=np.float64)
    w = _normalized_weights(weights, len(times))
    return [
        (t, float(w[np.isfinite(times) & (times <= t)].sum() * 100))
        for t in thresholds
    ]


def expected_response_time(
    min_times: np.ndarray,
    weights: np.ndarray,
    finite_only: bool = False,
) -> float:
    """Expected minimum response time under incident probability weights."""
    times = np.asarray(min_times, dtype=np.float64)
    w = _normalized_weights(weights, len(times))
    if finite_only:
        mask = np.isfinite(times)
        mass = w[mask].sum()
        if mass <= 0:
            return float("inf")
        return float(np.sum(w[mask] * times[mask]) / mass)

    if np.any((~np.isfinite(times)) & (w > 0)):
        return float("inf")
    return float(np.sum(w * times))


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


def weighted_station_zones(
    travel_times: np.ndarray,
    min_times: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Responsibility zones weighted by incident probability."""
    assignments = np.argmin(travel_times, axis=0)
    assignments[~np.isfinite(min_times)] = -1

    w = _normalized_weights(weights, travel_times.shape[1])
    zone_weights = []
    for s in range(travel_times.shape[0]):
        zone_weights.append((s, float(w[assignments == s].sum())))
    zone_weights.sort(key=lambda x: x[1], reverse=True)

    return assignments, zone_weights


def blind_spots(
    min_times: np.ndarray,
    threshold_min: float = 20.0,
) -> np.ndarray:
    """Indices of cells with response time above threshold (or unreachable)."""
    return np.where((min_times > threshold_min) | ~np.isfinite(min_times))[0]


def _normalized_weights(weights: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError("weights must have the same length as min_times")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    return w / total

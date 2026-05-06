"""Placement optimization module.

Public API:
- Objective: SeparableObjective + factories (mean_response_time, weighted_coverage,
  expected_failure)
- PlacementSet + builders (from_stations, attach_travel_times)
- sample_shore_candidates
- Problem
- Solution + algorithms (greedy, local_swap, greedy_then_swap)
"""

from .algorithms import Solution, greedy, greedy_then_swap, local_swap
from .candidates import sample_shore_candidates, sample_shore_points
from .objective import (
    SeparableObjective,
    expected_failure,
    mean_response_time,
    weighted_coverage,
)
from .placement import PlacementSet, attach_travel_times, from_stations
from .problem import Problem

__all__ = [
    "SeparableObjective",
    "mean_response_time",
    "weighted_coverage",
    "expected_failure",
    "PlacementSet",
    "from_stations",
    "attach_travel_times",
    "sample_shore_candidates",
    "sample_shore_points",
    "Problem",
    "Solution",
    "greedy",
    "local_swap",
    "greedy_then_swap",
]

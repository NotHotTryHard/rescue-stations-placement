"""Optimization problem: existing stations + candidates + objective + how many to add."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .objective import SeparableObjective
from .placement import PlacementSet


@dataclass
class Problem:
    existing: PlacementSet     # fixed ψ⁰
    candidates: PlacementSet   # 𝒞 — choose m of these
    objective: SeparableObjective
    m: int

    def __post_init__(self):
        if self.existing.travel_times.shape[1] != self.candidates.travel_times.shape[1]:
            raise ValueError("existing and candidates must share the same grid (N)")
        if self.objective.weights.shape[0] != self.candidates.travel_times.shape[1]:
            raise ValueError("objective.weights length must match grid size N")
        if self.m < 0 or self.m > self.candidates.K:
            raise ValueError(f"m must be in [0, K]; got m={self.m}, K={self.candidates.K}")

    @cached_property
    def base_field(self) -> np.ndarray:
        """t⁰ — minimum arrival time over all existing stations."""
        if self.existing.K == 0:
            return np.full(self.candidates.N, np.inf, dtype=np.float64)
        return self.existing.travel_times.min(axis=0)

    @property
    def base_value(self) -> float:
        return self.objective.value(self.base_field)

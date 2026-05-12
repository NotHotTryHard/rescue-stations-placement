"""Per-cell separable objectives.

All objectives are normalized to MINIMIZATION. A separable objective has the
form F(t) = Σ_j w_j · φ(t_j), with w summing to 1 and φ defined on [0, ∞].

Key consequence: marginal gain of adding a candidate c to a current set A
(field t_A, candidate times τ_c) is

    Δ(c | A) = Σ_j w_j (φ(t_A_j) - φ(min(t_A_j, τ_cj))) ≥ 0,

because adding a station can only decrease t and we choose φ non-decreasing.
This makes greedy submodular and allows fully vectorized marginal gains.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SeparableObjective:
    """Generic Σ w_j φ(t_j) objective with vectorized marginal gain."""

    weights: np.ndarray
    phi: Callable[[np.ndarray], np.ndarray]
    name: str = "objective"

    def __post_init__(self):
        w = np.asarray(self.weights, dtype=np.float64)
        if w.ndim != 1 or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
            raise ValueError("weights must be 1D, nonnegative, finite, with positive sum")
        self.weights = w / w.sum()

    def value(self, t: np.ndarray) -> float:
        return float(self.weights @ self.phi(np.asarray(t, dtype=np.float64)))

    def marginal_gain(self, t_A: np.ndarray, T_C: np.ndarray) -> np.ndarray:
        """Δ(c | A) for every row c of T_C. Returns (K,)."""
        t_A = np.asarray(t_A, dtype=np.float64)
        T_C = np.asarray(T_C, dtype=np.float64)
        phi_old = self.phi(t_A)
        # φ broadcasted over the (K, N) matrix of new fields min(T_C, t_A)
        phi_new = self.phi(np.minimum(T_C, t_A))
        return ((phi_old - phi_new) * self.weights).sum(axis=1)


def mean_response_time(weights: np.ndarray, t_cap_min: float = 120.0) -> SeparableObjective:
    """E[T] under Q. Caps unreachable cells at t_cap_min so the functional stays finite."""
    cap = float(t_cap_min)

    def phi(t):
        return np.minimum(t, cap)

    return SeparableObjective(weights=weights, phi=phi, name=f"mean_time(cap={cap:g})")


def weighted_coverage(weights: np.ndarray, threshold_min: float) -> SeparableObjective:
    """Negative weighted coverage: minimizing this maximizes Σ w_j 1{t_j ≤ T}."""
    T = float(threshold_min)

    def phi(t):
        return -(t <= T).astype(np.float64)

    return SeparableObjective(weights=weights, phi=phi, name=f"-coverage(T={T:g})")


def survival_exponential(median_min: float) -> Callable[[np.ndarray], np.ndarray]:
    """Exponential survival curve with constant hazard and S(median)=0.5."""
    median = max(float(median_min), 1e-9)
    lam = np.log(2.0) / median
    return lambda t: np.exp(-lam * np.asarray(t, dtype=np.float64))


def survival_increasing_intensity(
    median_min: float,
    max_time_min: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Survival curve with increasing hazard and S(max_time)=0."""
    median = max(float(median_min), 1e-9)
    max_time = max(float(max_time_min), median + 1e-9)
    scale = np.log(2.0) * (max_time - median) / (median * median)

    def survival(t):
        values = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(values)
        active = values < max_time
        clipped = np.maximum(values[active], 0.0)
        hazard = scale * clipped * clipped / np.maximum(max_time - clipped, 1e-9)
        out[active] = np.exp(-hazard)
        return out

    return survival


def expected_failure(weights: np.ndarray, survival: Callable[[np.ndarray], np.ndarray]) -> SeparableObjective:
    """E[1 - S(T)]. Survival must satisfy S(0)=1, monotone decreasing, S(inf)=0."""

    def phi(t):
        out = np.ones_like(t, dtype=np.float64)
        finite = np.isfinite(t)
        out[finite] = 1.0 - survival(t[finite])
        return out

    return SeparableObjective(weights=weights, phi=phi, name="expected_failure")

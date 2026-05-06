"""Greedy + local 1-swap algorithms for the placement problem."""

import time
from dataclasses import dataclass, field

import numpy as np

from .problem import Problem


@dataclass
class Solution:
    selected: np.ndarray                # (m,) indices into problem.candidates
    objective_value: float
    objective_history: list[float]      # F(base), F(after step 1), ..., F(after last step)
    final_field: np.ndarray             # (N,) t(A ∪ existing)
    iterations: int
    converged: bool
    meta: dict = field(default_factory=dict)


def _field_with(t_base: np.ndarray, T_C: np.ndarray, sel: np.ndarray) -> np.ndarray:
    if len(sel) == 0:
        return t_base.copy()
    return np.minimum(t_base, T_C[sel].min(axis=0))


def greedy(problem: Problem) -> Solution:
    """Submodular-style greedy: at each step pick candidate with the largest Δ."""
    t0 = time.perf_counter()
    obj = problem.objective
    T_C = problem.candidates.travel_times
    K = problem.candidates.K
    m = problem.m

    t_curr = problem.base_field.copy()
    f_curr = obj.value(t_curr)
    history = [f_curr]

    selected: list[int] = []
    available = np.ones(K, dtype=bool)

    for step in range(m):
        gains = obj.marginal_gain(t_curr, T_C)
        gains[~available] = -np.inf
        c_star = int(np.argmax(gains))
        if not available[c_star] or gains[c_star] <= 0:
            # No improving candidate (or all non-positive — happens for indicator phi
            # once nothing further can be saved). Stop early.
            break
        selected.append(c_star)
        available[c_star] = False
        t_curr = np.minimum(t_curr, T_C[c_star])
        f_curr = obj.value(t_curr)
        history.append(f_curr)

    return Solution(
        selected=np.asarray(selected, dtype=np.int64),
        objective_value=f_curr,
        objective_history=history,
        final_field=t_curr,
        iterations=len(selected),
        converged=(len(selected) == m),
        meta={"algorithm": "greedy", "elapsed_sec": time.perf_counter() - t0},
    )


def local_swap(problem: Problem, init: Solution, max_iters: int = 50) -> Solution:
    """First-improvement 1-swap: try replacing each a∈A with each c∈𝒞∖A."""
    t0 = time.perf_counter()
    obj = problem.objective
    T_C = problem.candidates.travel_times
    K = problem.candidates.K

    A = list(int(i) for i in init.selected)
    if len(A) == 0:
        return init

    t_curr = init.final_field.copy()
    f_curr = init.objective_value
    history = list(init.objective_history)

    swaps = 0
    converged = False
    for it in range(max_iters):
        improved = False
        in_A = np.zeros(K, dtype=bool)
        in_A[A] = True

        for ai in range(len(A)):
            # Field if we drop A[ai]: recompute min over (existing ∪ A\{a})
            kept = [A[j] for j in range(len(A)) if j != ai]
            t_without = _field_with(problem.base_field, T_C, np.asarray(kept, dtype=np.int64))
            # Vectorized: try every c not currently in A
            T_try = np.minimum(t_without, T_C)        # (K, N)
            f_try = (obj.phi(T_try) * obj.weights).sum(axis=1)  # (K,)
            f_try[in_A] = np.inf                       # cannot swap-in something already in A
            c_best = int(np.argmin(f_try))
            if f_try[c_best] + 1e-12 < f_curr:
                # Accept swap
                A[ai] = c_best
                in_A = np.zeros(K, dtype=bool)
                in_A[A] = True
                t_curr = np.minimum(t_without, T_C[c_best])
                f_curr = float(f_try[c_best])
                history.append(f_curr)
                swaps += 1
                improved = True
                break  # restart outer scan after first improvement

        if not improved:
            converged = True
            break

    return Solution(
        selected=np.asarray(A, dtype=np.int64),
        objective_value=f_curr,
        objective_history=history,
        final_field=t_curr,
        iterations=swaps,
        converged=converged,
        meta={
            "algorithm": "local_swap",
            "swaps": swaps,
            "max_iters": max_iters,
            "elapsed_sec": time.perf_counter() - t0,
        },
    )


def greedy_then_swap(problem: Problem, max_iters: int = 50) -> Solution:
    """Greedy seed + local 1-swap polishing."""
    t0 = time.perf_counter()
    seed = greedy(problem)
    polished = local_swap(problem, seed, max_iters=max_iters)
    polished.meta = {
        "algorithm": "greedy_then_swap",
        "greedy": seed.meta,
        "swap": polished.meta,
        "elapsed_sec": time.perf_counter() - t0,
    }
    polished.objective_history = seed.objective_history + polished.objective_history[len(seed.objective_history):]
    return polished

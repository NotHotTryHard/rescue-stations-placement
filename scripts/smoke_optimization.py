"""End-to-end smoke run of src/optimization on real project data.

Usage:
    uv run python scripts/smoke_optimization.py [--cell-size 200] [--shore-step 500] [-m 2]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    classify_cells_by_zone,
    get_passage_coords,
    load_risk_scenarios,
    load_shoreline,
    load_stations,
    load_water_polygon,
)
from src.graph import build_graph
from src.grid import generate_grid
from src.risk_distribution import IncidentDistribution
from src.optimization import (
    Problem,
    expected_failure,
    from_stations,
    greedy_then_swap,
    mean_response_time,
    sample_shore_candidates,
    weighted_coverage,
)


def survival_exponential(median_min: float = 10.0):
    lam = np.log(2.0) / median_min  # S(median) = 1/2
    return lambda t: np.exp(-lam * t)


def stage(label: str, fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    print(f"  [{time.perf_counter() - t:6.2f}s] {label}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-size", type=int, default=200, help="Grid cell size, m")
    ap.add_argument("--shore-step", type=float, default=500.0, help="Shore sampling step, m")
    ap.add_argument("--candidate-speed", type=float, default=40.0, help="km/h for new candidates")
    ap.add_argument("-m", type=int, default=2, help="Number of stations to add")
    ap.add_argument("--scenario", type=str, default="summer", help="Risk scenario name")
    args = ap.parse_args()

    print(f"# Pipeline (cell={args.cell_size} m, shore_step={args.shore_step} m, m={args.m})")
    t_total = time.perf_counter()

    water = stage("load_water_polygon", load_water_polygon)
    lats, lons, dlat, dlon = stage(
        "generate_grid", generate_grid, water, cell_size_m=args.cell_size
    )
    print(f"      grid cells N={len(lats)}")

    zones = stage("classify_cells_by_zone", classify_cells_by_zone, lats, lons)
    graph = stage(
        "build_graph", build_graph, lats, lons, dlat, dlon,
        cell_zones=zones, passage_coords=get_passage_coords(),
    )
    print(f"      graph edges = {graph.nnz}")

    stations = load_stations()
    existing = stage("travel times for existing stations", from_stations,
                     stations, graph=graph, grid_lats=lats, grid_lons=lons)

    candidates = stage(
        "sample shore candidates + travel times",
        sample_shore_candidates,
        step_m=args.shore_step,
        speed_kmh=args.candidate_speed,
        graph=graph,
        grid_lats=lats,
        grid_lons=lons,
        exclude_grid_indices=existing.grid_index,
    )
    print(f"      K candidates = {candidates.K}")

    scenarios = load_risk_scenarios()
    dist = stage(
        "build risk distribution",
        IncidentDistribution.from_scenario,
        args.scenario, lats, lons, scenarios,
        water_polygon=water, shoreline=load_shoreline(),
    )

    objectives = {
        "mean_time": mean_response_time(dist.weights, t_cap_min=120.0),
        "coverage_15min": weighted_coverage(dist.weights, threshold_min=15.0),
        "expected_failure": expected_failure(dist.weights, survival_exponential(10.0)),
    }

    print()
    for name, obj in objectives.items():
        print(f"## {name}  (φ={obj.name})")
        problem = Problem(existing=existing, candidates=candidates, objective=obj, m=args.m)
        F0 = problem.base_value
        sol = greedy_then_swap(problem)
        F1 = sol.objective_value
        delta = F0 - F1
        sign = "↓" if delta > 0 else ("=" if abs(delta) < 1e-12 else "↑")
        print(f"  F(base)        = {F0:+.6f}")
        print(f"  F(after opt)   = {F1:+.6f}  ({sign} {abs(delta):.6f})")
        print(f"  history        = {[f'{v:+.4f}' for v in sol.objective_history]}")
        print(f"  selected       = {sol.selected.tolist()}")
        for idx in sol.selected:
            la, lo = candidates.lat[idx], candidates.lon[idx]
            print(f"    {candidates.labels[idx]:<14s}  lat={la:.5f}  lon={lo:.5f}")
        print(f"  meta           = {sol.meta}")
        print()

    print(f"# total {time.perf_counter() - t_total:.2f}s")


if __name__ == "__main__":
    main()

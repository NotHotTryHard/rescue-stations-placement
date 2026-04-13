"""Shared session state: grid + travel times computed once, stored in st.session_state."""

import numpy as np
import streamlit as st

from .data import (
    load_water_polygon, load_stations, classify_cells_by_zone, get_passage_coords,
)
from .config import get_neighbor_offsets, get_config_value
from .grid import generate_grid, snap_to_grid
from .graph import build_graph
from .routing import compute_travel_times


def _compute(cell_size_m: int, neighbor_offsets: list[tuple[int, int]], neighbor_level: int):
    water = load_water_polygon()
    lats, lons, dlat, dlon = generate_grid(water, cell_size_m=cell_size_m)

    zones = classify_cells_by_zone(lats, lons)
    graph = build_graph(
        lats, lons, dlat, dlon,
        neighbor_offsets=neighbor_offsets,
        cell_zones=zones,
        passage_coords=get_passage_coords(),
        passage_radius_m=1000.0,
    )

    stations = load_stations()
    sources = [snap_to_grid(s.lat, s.lon, lats, lons) for s in stations]
    speeds = [s.speed_kmh for s in stations]
    travel_times = compute_travel_times(graph, sources, speeds)
    min_times = np.min(travel_times, axis=0)

    return {
        "cell_size": cell_size_m,
        "lats": lats,
        "lons": lons,
        "travel_times": travel_times,
        "min_times": min_times,
        "stations": stations,
        "neighbor_level": neighbor_level,
        "neighbor_offsets": tuple(neighbor_offsets),
    }


def sidebar_controls():
    """Shared cell size slider. Persists via session_state."""
    if "cell_size" not in st.session_state:
        st.session_state["cell_size"] = 300
    val = st.sidebar.slider(
        "Размер ячейки (м)", 40, 1000,
        value=st.session_state["cell_size"], step=20,
    )
    st.session_state["cell_size"] = val
    return val


def get_results():
    """Return precomputed results, recomputing only when cell_size changes."""
    cell_size = st.session_state.get("cell_size", 300)
    neighbor_level = int(get_config_value("neighbor_level"))
    neighbor_offsets = get_neighbor_offsets()
    neighbor_offsets_sig = tuple(neighbor_offsets)
    cached = st.session_state.get("results")

    if (
        cached is None
        or cached["cell_size"] != cell_size
        or cached.get("neighbor_level") != neighbor_level
        or cached.get("neighbor_offsets") != neighbor_offsets_sig
    ):
        with st.spinner("Расчёт сетки и маршрутов..."):
            st.session_state["results"] = _compute(cell_size, neighbor_offsets, neighbor_level)

    r = st.session_state["results"]
    return r["lats"], r["lons"], r["travel_times"], r["min_times"], r["stations"]


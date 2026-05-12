"""Shared session state: grid + travel times computed once, stored in st.session_state."""

import numpy as np
import streamlit as st

from .data import (
    Station,
    load_water_polygon, load_stations_raw, classify_cells_by_zone, get_passage_coords,
    load_risk_scenarios, load_shoreline,
)
from .config import ensure_config, get_neighbor_offsets, get_config_value
from .grid import generate_grid, snap_to_grid
from .graph import build_graph
from .risk_distribution import IncidentDistribution
from .routing import compute_travel_times

STATIONS_STATE_KEY = "active_stations"


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

    stations = get_active_stations()
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


def sidebar_section(title: str, expanded: bool = True):
    """Named sidebar group for related page parameters."""
    return st.sidebar.expander(title, expanded=expanded)


def sidebar_controls(container=None):
    """Shared cell size slider. Persists via session_state."""
    if container is None:
        container = st.sidebar
    if "cell_size" not in st.session_state:
        st.session_state["cell_size"] = 200
    val = container.slider(
        "Размер ячейки (м)", 40, 1000,
        value=st.session_state["cell_size"], step=20,
    )
    st.session_state["cell_size"] = val
    return val


def _invalidate_station_results():
    st.session_state.pop("results", None)


def _normalize_station_row(row: dict, fallback_id: str) -> dict:
    raw_name = row.get("name")
    raw_id = row.get("id")
    name = "" if raw_name is None else str(raw_name).strip()
    station_id = "" if raw_id is None else str(raw_id).strip()
    if not name or name.lower() == "nan":
        name = fallback_id
    if not station_id or station_id.lower() == "nan":
        station_id = name or fallback_id
    lat = float(row["lat"])
    lon = float(row["lon"])
    speed_kmh = float(row["speed_kmh"])
    if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(speed_kmh)):
        raise ValueError("Координаты и скорость должны быть конечными числами")
    return {
        "id": station_id,
        "name": name,
        "lat": lat,
        "lon": lon,
        "speed_kmh": speed_kmh,
    }


def _base_station_rows() -> list[dict]:
    return [
        _normalize_station_row(row, fallback_id=f"station_{i + 1}")
        for i, row in enumerate(load_stations_raw())
    ]


def _ensure_active_stations():
    if STATIONS_STATE_KEY in st.session_state:
        return
    rows = _base_station_rows()
    rows.extend(
        _normalize_station_row(row, fallback_id=f"added_{i + 1}")
        for i, row in enumerate(st.session_state.get("added_stations", []))
    )
    st.session_state[STATIONS_STATE_KEY] = rows


def get_added_stations_raw() -> list[dict]:
    """Stations added from optimization during the current Streamlit session."""
    _ensure_active_stations()
    base_ids = {row["id"] for row in _base_station_rows()}
    return [
        row.copy()
        for row in st.session_state[STATIONS_STATE_KEY]
        if row["id"] not in base_ids
    ]


def get_active_stations() -> list[Station]:
    """Stations active in the current session."""
    return [Station(**row) for row in get_active_stations_raw()]


def get_active_stations_raw() -> list[dict]:
    """Raw station dicts for pydeck layers, including session edits."""
    _ensure_active_stations()
    return [row.copy() for row in st.session_state[STATIONS_STATE_KEY]]


def set_active_stations_raw(rows: list[dict]):
    """Replace all active stations for the current session."""
    normalized = [
        _normalize_station_row(row, fallback_id=f"station_{i + 1}")
        for i, row in enumerate(rows)
    ]
    ids = [row["id"] for row in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("ID станций должны быть уникальными")
    if not normalized:
        raise ValueError("Нужна хотя бы одна станция")
    st.session_state[STATIONS_STATE_KEY] = normalized
    _invalidate_station_results()


def reset_active_stations():
    """Restore stations from data/stations.json for the current session."""
    st.session_state[STATIONS_STATE_KEY] = _base_station_rows()
    st.session_state.pop("added_stations", None)
    _invalidate_station_results()


def active_stations_signature() -> tuple:
    """Hashable signature for station-dependent caches."""
    return tuple(
        (s.id, round(float(s.lat), 7), round(float(s.lon), 7), float(s.speed_kmh))
        for s in get_active_stations()
    )


def add_session_station(name: str, lat: float, lon: float, speed_kmh: float, station_id: str) -> bool:
    """Add one station to the current session. Returns False if it already exists."""
    rows = get_active_stations_raw()
    ids = {row["id"] for row in rows}
    if station_id in ids:
        return False

    rows.append(
        {
            "id": station_id,
            "name": name,
            "lat": float(lat),
            "lon": float(lon),
            "speed_kmh": float(speed_kmh),
        }
    )
    set_active_stations_raw(rows)
    return True


def risk_scenario_control(cfg: dict, scenarios: dict, label: str = "Сценарий", container=None):
    """Shared risk scenario selector. Stores the selected key in cfg."""
    if container is None:
        container = st.sidebar
    options = list(scenarios)
    current = cfg.get("risk_scenario", "summer")
    if current not in scenarios:
        current = options[0]
    cfg["risk_scenario"] = container.selectbox(
        label,
        options,
        index=options.index(current),
        format_func=lambda key: scenarios[key].get("title", key),
    )
    return cfg["risk_scenario"]


def get_results():
    """Return precomputed results, recomputing only when cell_size changes."""
    cell_size = st.session_state.get("cell_size", 200)
    neighbor_level = int(get_config_value("neighbor_level"))
    neighbor_offsets = get_neighbor_offsets()
    neighbor_offsets_sig = tuple(neighbor_offsets)
    stations_sig = active_stations_signature()
    cached = st.session_state.get("results")

    if (
        cached is None
        or cached["cell_size"] != cell_size
        or cached.get("neighbor_level") != neighbor_level
        or cached.get("neighbor_offsets") != neighbor_offsets_sig
        or cached.get("stations_signature") != stations_sig
    ):
        with st.spinner("Расчёт сетки и маршрутов..."):
            st.session_state["results"] = _compute(cell_size, neighbor_offsets, neighbor_level)
            st.session_state["results"]["stations_signature"] = stations_sig

    r = st.session_state["results"]
    return r["lats"], r["lons"], r["travel_times"], r["min_times"], r["stations"]


def get_risk_distribution() -> IncidentDistribution:
    """Return the configured incident distribution over the current grid."""
    lats, lons, _, _, _ = get_results()
    cfg = ensure_config()
    scenarios = load_risk_scenarios()
    scenario = cfg.get("risk_scenario", "summer")
    if scenario not in scenarios:
        scenario = next(iter(scenarios))
        cfg["risk_scenario"] = scenario

    cell_size = st.session_state.get("cell_size", 200)
    cached = st.session_state.get("risk_distribution")
    signature = (scenario, cell_size, len(lats))
    if cached is None or cached.get("signature") != signature:
        with st.spinner("Расчёт модельной плотности происшествий..."):
            dist = IncidentDistribution.from_scenario(
                scenario,
                lats,
                lons,
                scenarios,
                water_polygon=load_water_polygon(),
                shoreline=load_shoreline(),
            )
            st.session_state["risk_distribution"] = {
                "signature": signature,
                "distribution": dist,
            }

    return st.session_state["risk_distribution"]["distribution"]

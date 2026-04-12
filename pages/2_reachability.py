"""Reachability heatmap: minimum time to reach each point from any station."""

import streamlit as st
import pydeck as pdk
import numpy as np

from src.data import (
    load_water_polygon, load_stations, load_stations_raw, load_passages,
    classify_cells_by_zone, get_passage_coords,
)
from src.grid import generate_grid, snap_to_grid
from src.graph import build_graph
from src.routing import compute_travel_times
from src.reachability import compute_reachability

st.set_page_config(page_title="Достижимость", layout="wide")
st.title("Карта достижимости")

# --- Sidebar controls ---
cell_size = st.sidebar.slider(
    "Размер ячейки сетки (м)", min_value=40, max_value=1000, value=300, step=20
)
max_time_display = st.sidebar.slider(
    "Макс. время на шкале (мин)", min_value=5, max_value=60, value=25, step=5
)


@st.cache_data(show_spinner="Генерация сетки...")
def get_grid(cell_size_m: int):
    water = load_water_polygon()
    lats, lons, dlat, dlon = generate_grid(water, cell_size_m=cell_size_m)
    return lats, lons, dlat, dlon


@st.cache_data(show_spinner="Построение графа и расчёт маршрутов...")
def get_reachability(cell_size_m: int):
    lats, lons, dlat, dlon = get_grid(cell_size_m)

    # Classify cells by zone and enforce passage constraints
    cell_zones = classify_cells_by_zone(lats, lons)
    passage_coords = get_passage_coords()
    graph = build_graph(
        lats, lons, dlat, dlon,
        cell_zones=cell_zones,
        passage_coords=passage_coords,
        passage_radius_m=1000.0,
    )

    stations = load_stations()
    source_indices = [snap_to_grid(s.lat, s.lon, lats, lons) for s in stations]
    speeds = [s.speed_kmh for s in stations]

    times = compute_travel_times(graph, source_indices, speeds)
    min_times = compute_reachability(times)

    return lats, lons, min_times


with st.spinner("Считаю..."):
    lats, lons, min_times = get_reachability(cell_size)

# --- Statistics ---
reachable_mask = np.isfinite(min_times)
reachable_times = min_times[reachable_mask]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ячеек в сетке", f"{len(lats):,}")
col2.metric("Достижимых", f"{reachable_mask.sum():,}")
col3.metric("Среднее время", f"{reachable_times.mean():.1f} мин" if len(reachable_times) else "—")
col4.metric("Макс. время", f"{reachable_times.max():.1f} мин" if len(reachable_times) else "—")


# --- Build heatmap data ---
def time_to_color(t: float, max_t: float) -> list[int]:
    """Green (fast) → Yellow → Red (slow). Unreachable = gray."""
    if not np.isfinite(t):
        return [128, 128, 128, 160]
    ratio = min(t / max_t, 1.0)
    if ratio < 0.5:
        # Green → Yellow
        r = int(255 * (ratio * 2))
        g = 220
    else:
        # Yellow → Red
        r = 255
        g = int(220 * (1 - (ratio - 0.5) * 2))
    return [r, g, 0, 180]


grid_data = []
for i in range(len(lats)):
    color = time_to_color(min_times[i], max_time_display)
    t = min_times[i] if np.isfinite(min_times[i]) else -1
    grid_data.append(
        {
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "color": color,
            "time_min": round(float(t), 1) if t >= 0 else "недостижимо",
        }
    )

# --- Layers ---
heatmap_layer = pdk.Layer(
    "ScatterplotLayer",
    data=grid_data,
    get_position="[lon, lat]",
    get_color="color",
    get_radius=cell_size * 0.6,
    pickable=True,
)

# Station markers on top
stations_raw = load_stations_raw()
station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=stations_raw,
    get_position="[lon, lat]",
    get_color=[0, 0, 0, 255],
    get_radius=250,
    pickable=True,
)
station_labels = pdk.Layer(
    "TextLayer",
    data=stations_raw,
    get_position="[lon, lat]",
    get_text="name",
    get_size=14,
    get_color=[0, 0, 0, 255],
    get_anchor="start",
    get_pixel_offset="[15, 0]",
)

# Passage markers
passages = load_passages()
passages_data = [
    {"name": v["name"], "lat": v["lat"], "lon": v["lon"]} for v in passages.values()
]
passage_layer = pdk.Layer(
    "ScatterplotLayer",
    data=passages_data,
    get_position="[lon, lat]",
    get_color=[255, 60, 0, 220],
    get_radius=150,
    pickable=True,
)

view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=0)

st.pydeck_chart(
    pdk.Deck(
        layers=[heatmap_layer, passage_layer, station_layer, station_labels],
        initial_view_state=view,
        tooltip={"text": "{name}\nВремя: {time_min} мин"},
    ),
    height=800,
)

# --- Legend ---
st.markdown(
    f"""
**Шкала цветов** (0 — {max_time_display} мин):
- :green[**Зелёный**] — быстрая достижимость (< {max_time_display // 3} мин)
- :orange[**Жёлтый**] — средняя ({max_time_display // 3}—{2 * max_time_display // 3} мин)
- :red[**Красный**] — медленная (> {2 * max_time_display // 3} мин)
- **Серый** — недостижимо
"""
)

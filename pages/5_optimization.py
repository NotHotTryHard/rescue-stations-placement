"""Доразмещение станций: жадный + локальный поиск над кандидатами с берега."""

import numpy as np
import pydeck as pdk
import streamlit as st

from src.config import MAPBOX_TOKEN, ensure_config, get_config_value, get_neighbor_offsets
from src.data import (
    classify_cells_by_zone,
    get_passage_coords,
    load_risk_scenarios,
    load_stations_raw,
    load_water_polygon,
)
from src.graph import build_graph
from src.grid import generate_grid
from src.optimization import (
    Problem,
    expected_failure,
    from_stations,
    greedy,
    greedy_then_swap,
    mean_response_time,
    sample_shore_candidates,
    weighted_coverage,
)
from src.session import get_results, get_risk_distribution, sidebar_controls

st.set_page_config(page_title="Оптимизация размещения", layout="wide")
st.title("Оптимизация: доразмещение станций")


@st.cache_resource(show_spinner=False)
def _build_graph_for(cell_size_m: int, neighbor_offsets_sig: tuple):
    """Reconstruct the graph used by session._compute (graph itself is not cached there)."""
    water = load_water_polygon()
    lats, lons, dlat, dlon = generate_grid(water, cell_size_m=cell_size_m)
    zones = classify_cells_by_zone(lats, lons)
    g = build_graph(
        lats, lons, dlat, dlon,
        neighbor_offsets=list(neighbor_offsets_sig),
        cell_zones=zones,
        passage_coords=get_passage_coords(),
        passage_radius_m=1000.0,
    )
    return g


@st.cache_resource(show_spinner=False)
def _build_candidates(cell_size_m: int, shore_step_m: float, speed_kmh: float,
                      neighbor_offsets_sig: tuple, exclude_indices: tuple):
    lats, lons, _, _, _ = get_results()
    graph = _build_graph_for(cell_size_m, neighbor_offsets_sig)
    return sample_shore_candidates(
        step_m=shore_step_m,
        speed_kmh=speed_kmh,
        graph=graph,
        grid_lats=lats,
        grid_lons=lons,
        exclude_grid_indices=list(exclude_indices),
    )


@st.cache_resource(show_spinner=False)
def _build_existing(cell_size_m: int, neighbor_offsets_sig: tuple):
    lats, lons, _, _, stations = get_results()
    graph = _build_graph_for(cell_size_m, neighbor_offsets_sig)
    return from_stations(stations, graph=graph, grid_lats=lats, grid_lons=lons)


# --- Sidebar ---
cfg = ensure_config()
cell_size = sidebar_controls()

risk_scenarios = load_risk_scenarios()
risk_options = list(risk_scenarios)
current_risk = cfg.get("risk_scenario", "summer")
if current_risk not in risk_scenarios:
    current_risk = risk_options[0]
cfg["risk_scenario"] = st.sidebar.selectbox(
    "Сценарий",
    risk_options,
    index=risk_options.index(current_risk),
    format_func=lambda key: risk_scenarios[key].get("title", key),
)

m = st.sidebar.slider("Сколько станций добавить (m)", 1, 8, 2)
shore_step = st.sidebar.slider("Шаг кандидатов вдоль берега (м)", 200, 2000, 500, 100)
candidate_speed = st.sidebar.slider("Скорость новых станций (км/ч)", 20, 80, 40, 5)

objective_choice = st.sidebar.radio(
    "Критерий оптимизации",
    ("mean_time", "coverage", "expected_failure"),
    format_func={
        "mean_time": "Среднее время прибытия",
        "coverage": "Покрытие за норматив",
        "expected_failure": "Ожидаемая невыживаемость",
    }.get,
)
coverage_T = st.sidebar.slider("Норматив для покрытия (мин)", 5, 40, 15, 1)
survival_median = st.sidebar.slider("Медиана выживаемости (мин)", 3, 30, 10, 1)
t_cap = st.sidebar.slider("Cap на время для среднего (мин)", 30, 240, 120, 10)

algorithm_choice = st.sidebar.radio(
    "Алгоритм",
    ("greedy", "greedy_then_swap"),
    format_func={"greedy": "Жадный", "greedy_then_swap": "Жадный + 1-swap"}.get,
)

# --- Compute ---
lats, lons, _, _, _ = get_results()
dist = get_risk_distribution()
neighbor_sig = tuple(get_neighbor_offsets())

with st.spinner("Подготовка существующих станций и кандидатов..."):
    existing = _build_existing(cell_size, neighbor_sig)
    candidates = _build_candidates(
        cell_size, float(shore_step), float(candidate_speed),
        neighbor_sig, tuple(int(i) for i in existing.grid_index),
    )

if objective_choice == "mean_time":
    objective = mean_response_time(dist.weights, t_cap_min=float(t_cap))
    fmt = lambda v: f"{v:+.3f} мин"
elif objective_choice == "coverage":
    objective = weighted_coverage(dist.weights, threshold_min=float(coverage_T))
    fmt = lambda v: f"{-v * 100:+.2f}%"  # show as positive coverage %
else:
    lam = np.log(2.0) / float(survival_median)
    objective = expected_failure(dist.weights, lambda t: np.exp(-lam * t))
    fmt = lambda v: f"{v * 100:+.2f}%"

problem = Problem(existing=existing, candidates=candidates, objective=objective, m=m)

with st.spinner(f"Оптимизация ({algorithm_choice})..."):
    if algorithm_choice == "greedy":
        solution = greedy(problem)
    else:
        solution = greedy_then_swap(problem)

# --- Metrics ---
F0 = problem.base_value
F1 = solution.objective_value
col1, col2, col3, col4 = st.columns(4)
col1.metric("Кандидатов", f"{candidates.K}")
col2.metric("F (база)", fmt(F0))
col3.metric("F (после opt)", fmt(F1), delta=fmt(F1 - F0), delta_color="inverse")
col4.metric("Время оптимизации", f"{solution.meta.get('elapsed_sec', 0):.3f} с")

# --- Map ---
view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=0)
map_style = get_config_value("map_style")

selected = solution.selected
selected_data = [
    {
        "lat": float(candidates.lat[i]),
        "lon": float(candidates.lon[i]),
        "label": candidates.labels[i],
        "rank": rank + 1,
    }
    for rank, i in enumerate(selected)
]

candidate_data = [
    {"lat": float(candidates.lat[i]), "lon": float(candidates.lon[i])}
    for i in range(candidates.K)
    if i not in set(int(s) for s in selected)
]

stations_raw = load_stations_raw()
station_charset = '"' + "".join(sorted({ch for s in stations_raw for ch in s["name"]})) + '"'

# Field difference: time saved per cell
saved = problem.base_field - solution.final_field
saved_max = max(float(saved.max()), 1e-9)
field_data = [
    {
        "lat": float(lats[j]),
        "lon": float(lons[j]),
        "color": [
            int(60 + 195 * min(saved[j] / saved_max, 1.0)),
            int(180 - 120 * min(saved[j] / saved_max, 1.0)),
            int(180 - 120 * min(saved[j] / saved_max, 1.0)),
            120,
        ],
        "saved_min": round(float(saved[j]), 2),
    }
    for j in range(len(lats))
]

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer", data=field_data, get_position="[lon, lat]",
                get_color="color", get_radius=cell_size * 0.6, pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer", data=candidate_data, get_position="[lon, lat]",
                get_color=[140, 140, 140, 100], get_radius=60, pickable=False,
            ),
            pdk.Layer(
                "ScatterplotLayer", data=stations_raw, get_position="[lon, lat]",
                get_color=[0, 0, 0, 255], get_radius=250, pickable=True,
            ),
            pdk.Layer(
                "TextLayer", data=stations_raw, get_position="[lon, lat]",
                get_text="name", character_set=station_charset, get_size=14,
                get_color=[0, 0, 0, 255], font_family='"Arial, sans-serif"',
                get_anchor="start", get_pixel_offset="[36, 0]",
            ),
            pdk.Layer(
                "ScatterplotLayer", data=selected_data, get_position="[lon, lat]",
                get_color=[20, 160, 60, 230], get_radius=320, pickable=True,
                stroked=True, get_line_color=[0, 0, 0, 255], line_width_min_pixels=2,
            ),
            pdk.Layer(
                "TextLayer", data=selected_data, get_position="[lon, lat]",
                get_text="rank", get_size=18,
                get_color=[255, 255, 255, 255],
                font_family='"Arial, sans-serif"',
                get_anchor="middle", get_alignment_baseline="center",
            ),
        ],
        initial_view_state=view,
        tooltip={"text": "Сэкономлено: {saved_min} мин\n{label}"},
        map_style=map_style,
        api_keys={"mapbox": MAPBOX_TOKEN},
    ),
    height=700,
)

st.caption(
    "Чёрные точки — действующие станции. Серые мелкие — кандидаты вдоль берега. "
    "Зелёные большие — выбранные оптимизатором (число = порядок добавления). "
    "Заливка ячеек — экономия времени прибытия после доразмещения (тёмно-розовый = больше)."
)

# --- Selected candidates table ---
st.subheader("Выбранные точки")
rows = []
for rank, idx in enumerate(selected, start=1):
    rows.append({
        "Порядок": rank,
        "Метка": candidates.labels[idx],
        "lat": f"{candidates.lat[idx]:.5f}",
        "lon": f"{candidates.lon[idx]:.5f}",
        "Скорость, км/ч": f"{candidates.speed_kmh[idx]:.0f}",
    })
st.dataframe(rows, hide_index=True, use_container_width=True)

# --- Objective history ---
st.subheader("Динамика функционала")
hist = solution.objective_history
hist_display = [-v if objective_choice == "coverage" else v for v in hist]
st.line_chart(
    {"F": hist_display},
    height=240,
)
st.caption(
    f"Итераций: {solution.iterations}, "
    f"сходимость: {'да' if solution.converged else 'нет'}, "
    f"мета: {solution.meta}"
)

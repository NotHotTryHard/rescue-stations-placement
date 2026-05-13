"""Доразмещение станций: жадный + локальный поиск над кандидатами с берега."""

import time

import numpy as np
import pydeck as pdk
import streamlit as st

from src.config import MAPBOX_TOKEN, ensure_config, get_config_value, get_neighbor_offsets
from src.data import (
    classify_cells_by_zone,
    get_passage_coords,
    load_risk_scenarios,
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
    survival_exponential,
    survival_increasing_intensity,
    weighted_coverage,
)
from src.session import (
    active_stations_signature,
    add_session_station,
    get_active_stations_raw,
    get_results,
    get_risk_distribution,
    risk_scenario_control,
    sidebar_controls,
    sidebar_section,
)

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
def _build_existing(cell_size_m: int, neighbor_offsets_sig: tuple, stations_sig: tuple):
    lats, lons, _, _, stations = get_results()
    graph = _build_graph_for(cell_size_m, neighbor_offsets_sig)
    return from_stations(stations, graph=graph, grid_lats=lats, grid_lons=lons)


# --- Sidebar ---
cfg = ensure_config()
risk_scenarios = load_risk_scenarios()

with sidebar_section("Сетка и данные"):
    cell_size = sidebar_controls(st)

with sidebar_section("Сценарий риска"):
    risk_scenario_control(cfg, risk_scenarios, container=st)

with sidebar_section("Кандидаты"):
    m = st.slider("Сколько станций добавить (m)", 1, 8, 2)
    shore_step = st.slider("Шаг кандидатов вдоль берега (м)", 200, 2000, 500, 100)
    candidate_speed = st.slider("Скорость новых станций (км/ч)", 20, 80, 40, 5)

with sidebar_section("Критерий"):
    objective_choice = st.radio(
        "Критерий оптимизации",
        ("mean_time", "coverage", "expected_failure"),
        format_func={
            "mean_time": "Среднее время прибытия",
            "coverage": "Покрытие за норматив",
            "expected_failure": "Ожидаемая невыживаемость",
        }.get,
    )

t_cap = 120
coverage_T = 15
survival_median = 10
survival_model = "increasing"
survival_max_time = 25
show_blind_spots = False
with sidebar_section("Выживаемость в воде"):
    survival_model = st.radio(
        "Критерий выживаемости",
        ("increasing", "exponential"),
        format_func={
            "increasing": "Возрастающая интенсивность",
            "exponential": "Экспонента",
        }.get,
        key="survival_model",
    )
    survival_median = st.slider(
        "Медиана выживаемости (мин)",
        3,
        30,
        10,
        1,
        key="survival_median_min",
    )
    if survival_model == "increasing":
        min_max_time = int(survival_median) + 1
        if "survival_max_time_min" not in st.session_state:
            st.session_state["survival_max_time_min"] = 25
        if st.session_state.get("survival_max_time_min", 25) < min_max_time:
            st.session_state["survival_max_time_min"] = min_max_time
        survival_max_time = st.slider(
            "Макс. значение выживаемости (мин)",
            min_max_time,
            120,
            key="survival_max_time_min",
        )
    coverage_T = st.slider(
        "Норматив для покрытия (мин)",
        5,
        40,
        15,
        1,
        key="coverage_threshold_min",
    )
    show_blind_spots = st.checkbox(
        "Показывать слепые пятна",
        value=False,
        key="show_blind_spots",
    )

with sidebar_section("Алгоритм"):
    algorithm_choice = st.radio(
        "Алгоритм",
        ("greedy", "greedy_then_swap"),
        index=1,
        format_func={"greedy": "Жадный", "greedy_then_swap": "Жадный + 1-swap"}.get,
    )

# --- Compute ---
lats, lons, _, _, _ = get_results()
dist = get_risk_distribution()
neighbor_sig = tuple(get_neighbor_offsets())
stations_sig = active_stations_signature()
total_t0 = time.perf_counter()

with st.spinner("Подготовка существующих станций и кандидатов..."):
    prep_t0 = time.perf_counter()
    existing = _build_existing(cell_size, neighbor_sig, stations_sig)
    candidates = _build_candidates(
        cell_size, float(shore_step), float(candidate_speed),
        neighbor_sig, tuple(int(i) for i in existing.grid_index),
    )
    prep_elapsed = time.perf_counter() - prep_t0

if objective_choice == "mean_time":
    objective = mean_response_time(dist.weights, t_cap_min=float(t_cap))
    fmt = lambda v: f"{v:+.3f} мин"
elif objective_choice == "coverage":
    objective = weighted_coverage(dist.weights, threshold_min=float(coverage_T))
    fmt = lambda v: f"{-v * 100:+.2f}%"  # show as positive coverage %
else:
    if survival_model == "exponential":
        survival = survival_exponential(float(survival_median))
    else:
        survival = survival_increasing_intensity(
            float(survival_median),
            float(survival_max_time),
        )
    objective = expected_failure(dist.weights, survival)
    fmt = lambda v: f"{v * 100:+.2f}%"

problem = Problem(existing=existing, candidates=candidates, objective=objective, m=m)

with st.spinner(f"Оптимизация ({algorithm_choice})..."):
    if algorithm_choice == "greedy":
        solution = greedy(problem)
    else:
        solution = greedy_then_swap(problem)
total_elapsed = time.perf_counter() - total_t0
algorithm_elapsed = float(solution.meta.get("elapsed_sec", 0.0))

# --- Metrics ---
F0 = problem.base_value
F1 = solution.objective_value
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Кандидатов", f"{candidates.K}")
col2.metric("F (база)", fmt(F0))
col3.metric("F (после opt)", fmt(F1), delta=fmt(F1 - F0), delta_color="inverse")
col4.metric("Время оптимизации", f"{total_elapsed:.3f} с")
col5.metric("Подготовка", f"{prep_elapsed:.3f} с")
col6.metric("Алгоритм", f"{algorithm_elapsed:.3f} с")

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

stations_raw = get_active_stations_raw()
station_charset = '"' + "".join(sorted({ch for s in stations_raw for ch in s["name"]})) + '"'

# Field layer: either remaining blind spots or time saved per cell.
if show_blind_spots:
    field_data = [
        {
            "lat": float(lats[j]),
            "lon": float(lons[j]),
            "color": [255, 0, 0, 190],
            "time_min": (
                round(float(solution.final_field[j]), 2)
                if np.isfinite(solution.final_field[j])
                else "недостижимо"
            ),
        }
        for j in range(len(lats))
        if not np.isfinite(solution.final_field[j]) or solution.final_field[j] > coverage_T
    ]
    field_tooltip = "Время: {time_min} мин\n{label}"
else:
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
    field_tooltip = "Сэкономлено: {saved_min} мин\n{label}"

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
        tooltip={"text": field_tooltip},
        map_style=map_style,
        api_keys={"mapbox": MAPBOX_TOKEN},
    ),
    height=700,
)

if show_blind_spots:
    st.caption(
        "Чёрные точки — действующие станции. Серые мелкие — кандидаты вдоль берега. "
        "Зелёные большие — выбранные оптимизатором. Красные ячейки — зоны вне норматива."
    )
else:
    st.caption(
        "Чёрные точки — действующие станции. Серые мелкие — кандидаты вдоль берега. "
        "Зелёные большие — выбранные оптимизатором (число = порядок добавления). "
        "Заливка ячеек — экономия времени прибытия после доразмещения (тёмно-розовый = больше)."
    )

# --- Selected candidates table ---
st.subheader("Выбранные точки")
active_station_ids = {
    row.get("id", row["name"])
    for row in get_active_stations_raw()
}
selected_rows = []
for rank, idx in enumerate(selected, start=1):
    label = str(candidates.labels[idx])
    selected_rows.append(
        {
            "rank": rank,
            "label": label,
            "station_id": f"opt_{label}",
            "station_name": f"Опт-{label.split('_')[-1]}",
            "lat": float(candidates.lat[idx]),
            "lon": float(candidates.lon[idx]),
            "speed_kmh": float(candidates.speed_kmh[idx]),
        }
    )

pending_rows = [
    row for row in selected_rows
    if row["station_id"] not in active_station_ids
]
if pending_rows and st.button("Добавить все выбранные станции"):
    for row in pending_rows:
        add_session_station(
            row["station_name"],
            row["lat"],
            row["lon"],
            row["speed_kmh"],
            row["station_id"],
        )
    _build_existing.clear()
    _build_candidates.clear()
    st.rerun()

header = st.columns([0.7, 1.4, 1.2, 1.2, 1.2, 1.5])
header[0].markdown("**Порядок**")
header[1].markdown("**Метка**")
header[2].markdown("**lat**")
header[3].markdown("**lon**")
header[4].markdown("**Скорость**")
header[5].markdown("**Действие**")

for station in selected_rows:
    row = st.columns([0.7, 1.4, 1.2, 1.2, 1.2, 1.5])
    row[0].write(station["rank"])
    row[1].write(station["label"])
    row[2].write(f"{station['lat']:.5f}")
    row[3].write(f"{station['lon']:.5f}")
    row[4].write(f"{station['speed_kmh']:.0f} км/ч")
    if station["station_id"] in active_station_ids:
        row[5].button("Добавлена", key=f"added_{station['station_id']}", disabled=True)
    elif row[5].button("Добавить станцию", key=f"add_{station['station_id']}"):
        add_session_station(
            station["station_name"],
            station["lat"],
            station["lon"],
            station["speed_kmh"],
            station["station_id"],
        )
        _build_existing.clear()
        _build_candidates.clear()
        st.rerun()

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

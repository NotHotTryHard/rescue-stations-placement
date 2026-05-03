"""Model incident risk density over the water grid."""

import numpy as np
import pydeck as pdk
import streamlit as st

from src.config import MAPBOX_TOKEN, ensure_config, get_config_value
from src.coverage import expected_response_time, weighted_coverage_at_thresholds
from src.data import load_risk_scenarios, load_stations_raw
from src.session import get_results, get_risk_distribution, sidebar_controls


def _scaled_values(values: np.ndarray, log_scale: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_value = values.max()
    if max_value <= 0:
        return np.zeros_like(values)
    ratios = values / max_value
    if log_scale:
        return np.log1p(ratios * 9999.0) / np.log1p(9999.0)
    return ratios


def _risk_color(value: float) -> list[int]:
    v = float(np.clip(value, 0.0, 1.0))
    return [
        int(50 + 205 * v),
        int(210 - 150 * v),
        int(60 - 45 * v),
        int(80 + 150 * v),
    ]


st.set_page_config(page_title="Плотность происшествий", layout="wide")
st.title("Модельная плотность происшествий")

cfg = ensure_config()
risk_scenarios = load_risk_scenarios()
risk_options = list(risk_scenarios)
current_risk = cfg.get("risk_scenario", "summer")
if current_risk not in risk_scenarios:
    current_risk = risk_options[0]

cell_size = sidebar_controls()
cfg["risk_scenario"] = st.sidebar.selectbox(
    "Сценарий",
    risk_options,
    index=risk_options.index(current_risk),
    format_func=lambda key: risk_scenarios[key].get("title", key),
)
value_mode = st.sidebar.radio(
    "Показатель",
    ["Интенсивность lambda", "Вероятность ячейки Q"],
)
log_scale = st.sidebar.checkbox("Логарифмическая окраска", value=True)
coverage_threshold = st.sidebar.slider("Порог покрытия риска (мин)", 5, 40, 20, 1)
show_samples = st.sidebar.checkbox("Показывать сэмплы происшествий", value=True)
sample_size = st.sidebar.slider("Число сэмплов", 50, 2000, 400, 50)
sample_seed = st.sidebar.number_input("Seed", value=1, step=1)

lats, lons, _, min_times, _ = get_results()
dist = get_risk_distribution()

values = dist.lambda_values if value_mode.startswith("Интенсивность") else dist.weights
plot_values = _scaled_values(values, log_scale=log_scale)

reachable_risk = dist.probability(np.isfinite(min_times)) * 100
covered_risk = (
    dist.probability(np.isfinite(min_times) & (min_times <= coverage_threshold)) * 100
)
mean_time = expected_response_time(min_times, dist.weights, finite_only=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ячеек", f"{len(lats):,}")
col2.metric("Достижимый риск", f"{reachable_risk:.1f}%")
col3.metric(f"Риск до {coverage_threshold} мин", f"{covered_risk:.1f}%")
col4.metric("Ожидаемое время", f"{mean_time:.1f} мин")

grid_data = [
    {
        "lat": float(lats[i]),
        "lon": float(lons[i]),
        "color": _risk_color(plot_values[i]),
        "lambda_value": f"{dist.lambda_values[i]:.3e}",
        "probability_pct": f"{dist.weights[i] * 100:.4f}",
    }
    for i in range(len(lats))
]

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=grid_data,
        get_position="[lon, lat]",
        get_color="color",
        get_radius=cell_size * 0.6,
        pickable=True,
    )
]

if show_samples:
    sample_lats, sample_lons = dist.sample_points(sample_size, rng=int(sample_seed))
    sample_data = [
        {"lat": float(lat), "lon": float(lon)}
        for lat, lon in zip(sample_lats, sample_lons)
    ]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=sample_data,
            get_position="[lon, lat]",
            get_color=[20, 30, 40, 180],
            get_radius=max(cell_size * 0.25, 35),
            pickable=False,
        )
    )

stations_raw = load_stations_raw()
station_charset = (
    '"' + "".join(sorted({ch for s in stations_raw for ch in s["name"]})) + '"'
)
layers.extend(
    [
        pdk.Layer(
            "ScatterplotLayer",
            data=stations_raw,
            get_position="[lon, lat]",
            get_color=[0, 0, 0, 255],
            get_radius=250,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=stations_raw,
            get_position="[lon, lat]",
            get_text="name",
            character_set=station_charset,
            font_family='"Arial, sans-serif"',
            get_size=14,
            get_color=[0, 0, 0, 255],
            get_anchor="start",
            get_pixel_offset="[36, 0]",
        ),
    ]
)

view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=0)
st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={"text": "lambda: {lambda_value}\nQ: {probability_pct}%"},
        map_style=get_config_value("map_style"),
        api_keys={"mapbox": MAPBOX_TOKEN},
    ),
    height=800,
)

rows = weighted_coverage_at_thresholds(min_times, dist.weights, [5, 10, 15, 20, 25, 30])
st.subheader("Покрытие по модельному риску")
cols = st.columns(len(rows))
for col, (threshold, pct) in zip(cols, rows):
    col.metric(f"{threshold:.0f} мин", f"{pct:.1f}%")

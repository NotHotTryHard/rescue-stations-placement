"""Model incident risk density over the water grid."""

import numpy as np
import pydeck as pdk
import streamlit as st

from src.config import MAPBOX_TOKEN, ensure_config, get_config_value
from src.coverage import weighted_coverage_at_thresholds
from src.data import load_risk_scenarios, load_stations_raw
from src.grid import METERS_PER_DEG_LAT, METERS_PER_DEG_LON
from src.session import (
    get_results,
    get_risk_distribution,
    risk_scenario_control,
    sidebar_controls,
    sidebar_section,
)


def _log_scale_factor(scenario: str) -> float:
    return 2999.0 if scenario == "winter" else 9.0


def _default_hex_elevation_scale(scenario: str) -> int:
    return 10000 if scenario == "winter" else 4000


def _max_hex_elevation_scale(scenario: str) -> int:
    return 20000 if scenario == "winter" else 10000


def _scaled_values(values: np.ndarray, log_scale: bool, log_factor: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_value = values.max()
    if max_value <= 0:
        return np.zeros_like(values)
    ratios = values / max_value
    if log_scale:
        return np.log1p(ratios * log_factor) / np.log1p(log_factor)
    return ratios


def _risk_color(value: float) -> list[int]:
    v = float(np.clip(value, 0.0, 1.0))
    return [
        int(50 + 205 * v),
        int(210 - 150 * v),
        int(60 - 45 * v),
        int(80 + 150 * v),
    ]


def _hex_round(q: float, r: float) -> tuple[int, int]:
    x, z = q, r
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)

    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def _hex_polygon(center_x_m: float, center_y_m: float, radius_m: float) -> list[list[float]]:
    coords = []
    for angle_deg in (0, 60, 120, 180, 240, 300):
        angle = np.deg2rad(angle_deg)
        lon = (center_x_m + radius_m * np.cos(angle)) / METERS_PER_DEG_LON
        lat = (center_y_m + radius_m * np.sin(angle)) / METERS_PER_DEG_LAT
        coords.append([float(lon), float(lat)])
    return coords


def _hex_tower_data(
    lats: np.ndarray,
    lons: np.ndarray,
    weights: np.ndarray,
    radius_m: float,
    elevation_scale: float,
    log_scale: bool,
    log_factor: float,
) -> list[dict]:
    size = float(radius_m)
    bins: dict[tuple[int, int], list[float]] = {}
    probabilities = np.asarray(weights, dtype=np.float64)
    max_probability = probabilities.max()
    if max_probability <= 0:
        return []
    relative_probabilities = probabilities / max_probability

    xs = np.asarray(lons, dtype=np.float64) * METERS_PER_DEG_LON
    ys = np.asarray(lats, dtype=np.float64) * METERS_PER_DEG_LAT
    origin_x = float(xs.min())
    origin_y = float(ys.min())
    for x, y, relative_probability, weight in zip(
        xs, ys, relative_probabilities, probabilities
    ):
        rel_x = x - origin_x
        rel_y = y - origin_y
        q = (2.0 / 3.0 * rel_x) / size
        r = (-1.0 / 3.0 * rel_x + np.sqrt(3.0) / 3.0 * rel_y) / size
        key = _hex_round(q, r)
        if key not in bins:
            bins[key] = [0.0, 0.0, 0.0, 0.0, 0.0]
        bins[key][0] += float(relative_probability)
        bins[key][1] += float(weight)
        bins[key][2] += float(x)
        bins[key][3] += float(y)
        bins[key][4] += 1.0

    towers = []
    for (q, r), values in bins.items():
        count = values[4]
        relative_probability_mean = values[0] / count
        center_x = origin_x + size * 1.5 * q
        center_y = origin_y + size * np.sqrt(3.0) * (r + q / 2.0)
        color_value = relative_probability_mean
        if log_scale:
            color_value = np.log1p(relative_probability_mean * log_factor) / np.log1p(log_factor)
        towers.append(
            {
                "polygon": _hex_polygon(center_x, center_y, size),
                "elevation": relative_probability_mean * elevation_scale,
                "color": _risk_color(color_value),
                "relative_probability": f"{relative_probability_mean:.3f}",
                "probability_pct": f"{values[1] * 100:.4f}",
            }
        )

    return towers


st.set_page_config(page_title="Плотность происшествий", layout="wide")
st.title("Модельная плотность происшествий")

cfg = ensure_config()
risk_scenarios = load_risk_scenarios()
with sidebar_section("Сетка и данные"):
    cell_size = sidebar_controls(st)

with sidebar_section("Сценарий риска"):
    risk_scenario_control(cfg, risk_scenarios, container=st)

with sidebar_section("Выживаемость и нормативы"):
    coverage_threshold = st.slider(
        "Норматив для покрытия (мин)",
        5,
        40,
        15,
        1,
        key="coverage_threshold_min",
    )

with sidebar_section("Визуализация"):
    log_scale = st.checkbox("Логарифмическая окраска", value=True)

with sidebar_section("Сэмплы происшествий"):
    show_samples = st.checkbox("Показывать сэмплы происшествий", value=True)
    sample_size = st.slider("Число сэмплов", 50, 2000, 400, 50)
    sample_seed = st.number_input("Seed", value=1, step=1)

with sidebar_section("3D-профиль"):
    show_hex_towers = st.checkbox("3D-гексагоны риска", value=True)
    hex_radius = st.slider("Радиус 3D-гексагона (м)", 150, 1200, 350, 50)
    hex_elevation_scale = st.slider(
        "Масштаб высоты 3D",
        300,
        _max_hex_elevation_scale(cfg["risk_scenario"]),
        _default_hex_elevation_scale(cfg["risk_scenario"]),
        100,
    )
    hex_pitch = st.slider("Начальный наклон 3D-карты", 0, 85, 55, 5)

lats, lons, _, min_times, _ = get_results()
dist = get_risk_distribution()

log_factor = _log_scale_factor(cfg["risk_scenario"])
plot_values = _scaled_values(dist.lambda_values, log_scale=log_scale, log_factor=log_factor)

covered_risk = (
    dist.probability(np.isfinite(min_times) & (min_times <= coverage_threshold)) * 100
)
mean_time = dist.expected_time(min_times, finite_only=True)

col1, col2, col3 = st.columns(3)
col1.metric("Ячеек", f"{len(lats):,}")
col2.metric(f"Риск до {coverage_threshold} мин", f"{covered_risk:.1f}%")
col3.metric("Ожидаемое время", f"{mean_time:.1f} мин")

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

if show_hex_towers:
    st.subheader("3D-профиль модельного риска")
    st.caption("Камеру можно вращать: Shift + стрелки.")
    hex_data = _hex_tower_data(
        lats,
        lons,
        dist.weights,
        radius_m=float(hex_radius),
        elevation_scale=float(hex_elevation_scale),
        log_scale=log_scale,
        log_factor=log_factor,
    )
    hex_layers = [
        pdk.Layer(
            "PolygonLayer",
            data=hex_data,
            get_polygon="polygon",
            get_fill_color="color",
            get_elevation="elevation",
            extruded=True,
            wireframe=True,
            pickable=True,
            opacity=0.78,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=stations_raw,
            get_position="[lon, lat]",
            get_color=[0, 0, 0, 255],
            get_radius=250,
            pickable=True,
        ),
    ]
    hex_view = pdk.ViewState(
        latitude=60.00,
        longitude=29.85,
        zoom=10,
        pitch=hex_pitch,
        bearing=-18,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=hex_layers,
            views=[
                pdk.View(
                    type="MapView",
                    controller={
                        "dragRotate": True,
                        "keyboard": True,
                        "minPitch": 0,
                        "maxPitch": 85,
                    },
                )
            ],
            initial_view_state=hex_view,
            tooltip={
                "text": "relative Q/cell: {relative_probability}\nQ в гексагоне: {probability_pct}%"
            },
            map_style=get_config_value("map_style"),
            api_keys={"mapbox": MAPBOX_TOKEN},
        ),
        height=760,
    )

view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=0)
st.subheader("Плоская карта плотности")
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

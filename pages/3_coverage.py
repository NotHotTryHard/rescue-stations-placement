"""Coverage analysis: cumulative coverage, station zones, blind spots."""

import streamlit as st
import pydeck as pdk
import numpy as np

from src.data import load_stations_raw
from src.session import sidebar_controls, get_results
from src.coverage import coverage_curve, coverage_at_thresholds, station_zones, blind_spots

st.set_page_config(page_title="Покрытие", layout="wide")
st.title("Анализ покрытия")

cell_size = sidebar_controls()
blind_threshold = st.sidebar.slider("Порог слепых пятен (мин)", 5, 40, 20, 1)

lats, lons, travel, min_times, stations = get_results()
stations_raw = load_stations_raw()
station_charset = '"' + "".join(sorted({ch for s in stations_raw for ch in s["name"]})) + '"'
view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=0)

# --- 1. Coverage curve ---
st.subheader("Кривая покрытия")

thresholds, pcts = coverage_curve(min_times, max_time=30.0, step=0.5)
st.line_chart(data={"Время (мин)": thresholds, "Покрытие (%)": pcts}, x="Время (мин)", y="Покрытие (%)")

# --- 2. Coverage at thresholds ---
st.subheader("Покрытие по порогам")

rows = coverage_at_thresholds(min_times, [5, 10, 15, 20, 25, 30])
cols = st.columns(len(rows))
for col, (t, pct) in zip(cols, rows):
    col.metric(f"{t:.0f} мин", f"{pct:.1f}%")

# --- 3. Station zones ---
st.subheader("Зоны ответственности")

assignments, zone_sizes = station_zones(travel, min_times)

COLORS = [
    [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
    [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
    [153, 153, 153], [0, 200, 200],
]

zone_data = [
    {"lat": float(lats[i]), "lon": float(lons[i]),
     "color": COLORS[assignments[i] % len(COLORS)] + [160],
     "name": stations[assignments[i]].name}
    for i in range(len(lats)) if assignments[i] >= 0
]

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer("ScatterplotLayer", data=zone_data, get_position="[lon, lat]",
                      get_color="color", get_radius=cell_size * 0.6, pickable=True),
            pdk.Layer("ScatterplotLayer", data=stations_raw, get_position="[lon, lat]",
                      get_color=[0, 0, 0, 255], get_radius=250, pickable=True),
            pdk.Layer("TextLayer", data=stations_raw, get_position="[lon, lat]",
                      get_text="name", character_set=station_charset, get_size=14, get_color=[0, 0, 0, 255],
                      font_family='"Arial, sans-serif"',
                      get_anchor="start", get_pixel_offset="[36, 0]"),
        ],
        initial_view_state=view,
        tooltip={"text": "Станция: {name}"},
        map_style="light",
    ),
    height=600,
)

for s_idx, count in zone_sizes:
    st.write(f"- **{stations[s_idx].name}**: {count:,} ({count / len(lats) * 100:.1f}%)")

# --- 4. Blind spots ---
st.subheader(f"Слепые пятна (>{blind_threshold} мин)")

spots = blind_spots(min_times, threshold_min=blind_threshold)

if len(spots) == 0:
    st.success(f"Вся акватория достижима за {blind_threshold} мин!")
else:
    st.warning(f"{len(spots):,} ячеек ({len(spots) / len(lats) * 100:.1f}%) с временем >{blind_threshold} мин")

    spot_data = [{"lat": float(lats[i]), "lon": float(lons[i]), "color": [255, 0, 0, 200]} for i in spots]

    st.pydeck_chart(
        pdk.Deck(
            layers=[
                pdk.Layer("ScatterplotLayer", data=spot_data, get_position="[lon, lat]",
                          get_color="color", get_radius=cell_size * 0.6),
                pdk.Layer("ScatterplotLayer", data=stations_raw, get_position="[lon, lat]",
                          get_color=[0, 0, 0, 255], get_radius=250, pickable=True),
                pdk.Layer("TextLayer", data=stations_raw, get_position="[lon, lat]",
                          get_text="name", character_set=station_charset, get_size=14, get_color=[0, 0, 0, 255],
                          font_family='"Arial, sans-serif"',
                          get_anchor="start", get_pixel_offset="[36, 0]"),
            ],
            initial_view_state=view,
            tooltip={"text": "{name}"},
            map_style="light",
        ),
        height=600,
    )

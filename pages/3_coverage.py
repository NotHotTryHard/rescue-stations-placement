"""Coverage analysis: coverage thresholds, station responsibility zones."""

import streamlit as st
import pydeck as pdk

from src.data import load_stations_raw
from src.config import get_config_value, MAPBOX_TOKEN
from src.session import sidebar_controls, get_results, get_risk_distribution
from src.coverage import (
    coverage_at_thresholds,
    station_zones,
    weighted_coverage_at_thresholds,
    weighted_station_zones,
)

st.set_page_config(page_title="Зоны ответственности", layout="wide")
st.title("Зоны ответственности")

cell_size = sidebar_controls()

lats, lons, travel, min_times, stations = get_results()
dist = get_risk_distribution()
stations_raw = load_stations_raw()
station_charset = '"' + "".join(sorted({ch for s in stations_raw for ch in s["name"]})) + '"'
view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=0)
map_style = get_config_value("map_style")

rows = coverage_at_thresholds(min_times, [5, 10, 15, 20, 25, 30])
cols = st.columns(len(rows))
for col, (t, pct) in zip(cols, rows):
    col.metric(f"{t:.0f} мин", f"{pct:.1f}%")

st.subheader("Покрытие по модельному риску")
risk_rows = weighted_coverage_at_thresholds(min_times, dist.weights, [5, 10, 15, 20, 25, 30])
risk_cols = st.columns(len(risk_rows))
for col, (t, pct) in zip(risk_cols, risk_rows):
    col.metric(f"{t:.0f} мин", f"{pct:.1f}%")

assignments, zone_sizes = station_zones(travel, min_times)
_, zone_risks = weighted_station_zones(travel, min_times, dist.weights)
risk_by_station = {s_idx: risk for s_idx, risk in zone_risks}

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
        map_style=map_style,
        api_keys={"mapbox": MAPBOX_TOKEN},
    ),
    height=800,
)

for s_idx, count in zone_sizes:
    risk_pct = risk_by_station.get(s_idx, 0.0) * 100
    st.write(
        f"- **{stations[s_idx].name}**: {count:,} "
        f"({count / len(lats) * 100:.1f}% площади), {risk_pct:.1f}% риска"
    )

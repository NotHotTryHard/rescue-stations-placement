"""Reachability heatmap: minimum time to reach each point from any station."""

import streamlit as st
import pydeck as pdk
import numpy as np

from src.data import load_stations_raw, load_passages
from src.session import sidebar_controls, get_results

st.set_page_config(page_title="Достижимость", layout="wide")
st.title("Карта достижимости")

cell_size = sidebar_controls()
max_time_display = st.sidebar.slider("Макс. время на шкале (мин)", 5, 60, 25, 5)

lats, lons, _, min_times, _ = get_results()

# --- Statistics ---
reachable = min_times[np.isfinite(min_times)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ячеек", f"{len(lats):,}")
col2.metric("Достижимых", f"{len(reachable):,}")
col3.metric("Среднее", f"{reachable.mean():.1f} мин" if len(reachable) else "—")
col4.metric("Максимум", f"{reachable.max():.1f} мин" if len(reachable) else "—")


# --- Heatmap ---
def time_to_color(t, max_t):
    if not np.isfinite(t):
        return [128, 128, 128, 160]
    ratio = min(t / max_t, 1.0)
    if ratio < 0.5:
        r, g = int(255 * ratio * 2), 220
    else:
        r, g = 255, int(220 * (1 - (ratio - 0.5) * 2))
    return [r, g, 0, 180]


grid_data = [
    {
        "lat": float(lats[i]),
        "lon": float(lons[i]),
        "color": time_to_color(min_times[i], max_time_display),
        "time_min": round(float(min_times[i]), 1) if np.isfinite(min_times[i]) else "недостижимо",
    }
    for i in range(len(lats))
]

stations_raw = load_stations_raw()
station_charset = '"' + "".join(sorted({ch for s in stations_raw for ch in s["name"]})) + '"'
passages = load_passages()
passages_data = [{"name": v["name"], "lat": v["lat"], "lon": v["lon"]} for v in passages.values()]

view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=0)

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer("ScatterplotLayer", data=grid_data, get_position="[lon, lat]",
                      get_color="color", get_radius=cell_size * 0.6, pickable=True),
            pdk.Layer("ScatterplotLayer", data=passages_data, get_position="[lon, lat]",
                      get_color=[255, 60, 0, 220], get_radius=150, pickable=True),
            pdk.Layer("ScatterplotLayer", data=stations_raw, get_position="[lon, lat]",
                      get_color=[0, 0, 0, 255], get_radius=250, pickable=True),
            pdk.Layer("TextLayer", data=stations_raw, get_position="[lon, lat]",
                      get_text="name", character_set=station_charset, get_size=14, get_color=[0, 0, 0, 255],
                      font_family='"Arial, sans-serif"',
                      get_anchor="start", get_pixel_offset="[36, 0]")
        ],
        initial_view_state=view,
        tooltip={"text": "{name}\nВремя: {time_min} мин"},
        map_style="light",
    ),
    height=800,
)

st.markdown(
    f"**Шкала:** :green[зелёный] (< {max_time_display // 3} мин) → "
    f":orange[жёлтый] → :red[красный] (> {2 * max_time_display // 3} мин) → серый (недостижимо)"
)

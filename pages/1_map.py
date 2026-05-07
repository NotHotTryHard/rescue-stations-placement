"""Base map: water zones, stations, and passages."""

import streamlit as st
import pydeck as pdk

from shapely.geometry import mapping

from src.data import (
    load_kronshtadt_outline,
    load_passages,
    load_shoreline,
    load_stations_raw,
    load_zones_geojson,
)
from src.config import get_config_value, MAPBOX_TOKEN

st.set_page_config(page_title="Карта", layout="wide")
st.title("Карта акватории")


@st.cache_data
def get_data():
    zones = load_zones_geojson()
    passages = load_passages()
    stations = load_stations_raw()
    mainland_geom = mapping(load_shoreline())
    kron_geom = mapping(load_kronshtadt_outline())
    return zones, passages, stations, mainland_geom, kron_geom


zones, passages, stations, mainland_geom, kron_geom = get_data()
station_charset = '"' + "".join(sorted({ch for s in stations for ch in s["name"]})) + '"'

zones_colored = {
    "type": "FeatureCollection",
    "features": [
        {
            **f,
            "properties": {
                **f["properties"],
                "fill_color": [0, 80, 200, 100]
                if f["properties"]["zone"] == "north"
                else [0, 160, 80, 100],
                "line_color": [0, 40, 150, 220]
                if f["properties"]["zone"] == "north"
                else [0, 100, 40, 220],
            },
        }
        for f in zones["features"]
    ],
}

passages_data = [
    {"name": v["name"], "lat": v["lat"], "lon": v["lon"]} for v in passages.values()
]

zone_layer = pdk.Layer(
    "GeoJsonLayer",
    data=zones_colored,
    filled=True,
    stroked=True,
    get_fill_color="properties.fill_color",
    get_line_color="properties.line_color",
    get_line_width=30,
    pickable=True,
)

mainland_layer = pdk.Layer(
    "GeoJsonLayer",
    data={"type": "Feature", "geometry": mainland_geom, "properties": {"name": "Берег материка"}},
    stroked=True,
    filled=False,
    get_line_color=[230, 30, 60, 80],    # magenta-red, translucent
    get_line_width=70,
    line_width_min_pixels=2,
    pickable=True,
)

kronshtadt_layer = pdk.Layer(
    "GeoJsonLayer",
    data={"type": "Feature", "geometry": kron_geom, "properties": {"name": "Берег Кронштадта"}},
    stroked=True,
    filled=False,
    get_line_color=[20, 200, 230, 80],   # cyan, translucent
    get_line_width=70,
    line_width_min_pixels=2,
    pickable=True,
)

passage_layer = pdk.Layer(
    "ScatterplotLayer",
    data=passages_data,
    get_position="[lon, lat]",
    get_color=[255, 60, 0, 220],
    get_radius=150,
    pickable=True,
)

station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=stations,
    get_position="[lon, lat]",
    get_color=[255, 200, 0, 240],
    get_radius=200,
    pickable=True,
)

station_labels = pdk.Layer(
    "TextLayer",
    data=stations,
    get_position="[lon, lat]",
    get_text="name",
    character_set=station_charset,
    font_family='"Arial, sans-serif"',
    get_size=14,
    get_color=[0, 0, 0, 255],
    get_anchor="start",
    get_pixel_offset="[36, 0]",
)

view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=40)
map_style = get_config_value("map_style")

st.pydeck_chart(
    pdk.Deck(
        layers=[zone_layer, mainland_layer, kronshtadt_layer, passage_layer, station_layer, station_labels],
        initial_view_state=view,
        tooltip={"text": "{name}"},
        map_style=map_style,
        api_keys={"mapbox": MAPBOX_TOKEN},
    ),
    height=800,
)

st.caption(
    "🟥 Берег материка (`shoreline.geojson`) — источник кандидатов «материк». "
    "🟦 Контур Кронштадта (включая прилегающие сегменты КЗС) — источник кандидатов «остров». "
    "Тёмные обводы зон — полные границы акватории (туда станции уже не сэмплируются)."
)

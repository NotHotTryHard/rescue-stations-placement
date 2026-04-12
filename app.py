import json
from pathlib import Path

import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Финский залив", layout="wide")
st.title("Финский залив")

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    with open(DATA_DIR / "neva_zone.geojson") as f:
        zones = json.load(f)
    with open(DATA_DIR / "passages.json") as f:
        passages = json.load(f)
    return zones, passages

zones, passages = load_data()

with open(DATA_DIR / "stations.json") as f:
    stations = json.load(f)

def zone_color(feature):
    return [0, 80, 200, 100] if feature["properties"]["zone"] == "north" else [0, 160, 80, 100]

zones_colored = {
    "type": "FeatureCollection",
    "features": [
        {**f, "properties": {**f["properties"],
            "fill_color": [0, 80, 200, 100] if f["properties"]["zone"] == "north" else [0, 160, 80, 100],
            "line_color": [0, 40, 150, 220] if f["properties"]["zone"] == "north" else [0, 100, 40, 220],
        }}
        for f in zones["features"]
    ]
}

passages_data = [
    {"name": v["name"], "lat": v["lat"], "lon": v["lon"]}
    for v in passages.values()
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
    get_size=14,
    get_color=[0, 0, 0, 255],
    get_anchor="start",
    get_pixel_offset="[15, 0]",
)

view = pdk.ViewState(latitude=60.00, longitude=29.85, zoom=10, pitch=40)

st.pydeck_chart(
    pdk.Deck(
        layers=[zone_layer, passage_layer, station_layer, station_labels],
        initial_view_state=view,
        tooltip={"text": "{name}"},
    ),
    height=800,
)

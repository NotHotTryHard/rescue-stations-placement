"""Session configuration page."""

import streamlit as st

from src.config import DEFAULT_CONFIG, MAP_STYLE_OPTIONS, ensure_config

st.set_page_config(page_title="Настройки", layout="wide")
st.title("Настройки")

cfg = ensure_config()

map_style = cfg.get("map_style", DEFAULT_CONFIG["map_style"])
if map_style not in MAP_STYLE_OPTIONS:
    map_style = DEFAULT_CONFIG["map_style"]

cfg["map_style"] = st.selectbox(
    "Стиль карты (pydeck map_style)",
    MAP_STYLE_OPTIONS,
    index=MAP_STYLE_OPTIONS.index(map_style),
)

if st.button("Сбросить настройки текущей сессии"):
    st.session_state["app_config"] = DEFAULT_CONFIG.copy()
    st.rerun()

st.caption("Стили mapbox:// могут требовать токен Mapbox в окружении.")


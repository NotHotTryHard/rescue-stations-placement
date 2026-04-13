"""Session-scoped app configuration with project defaults."""

import streamlit as st

MAPBOX_TOKEN = "pk.eyJ1Ijoibm90aG90dHJ5aGFyZCIsImEiOiJjbW54bWg5aWMwM2FxMnFyOHlkeTJ1ZG5pIn0.AiLtFFbOXt3MIouqO-cUag"  # public token for mapbox, good for commiting

DEFAULT_CONFIG = {
    "map_style": "mapbox://styles/mapbox/light-v11",
}

MAP_STYLE_OPTIONS = [
    "mapbox://styles/mapbox/light-v11",
    "mapbox://styles/mapbox/dark-v11",
    "mapbox://styles/mapbox/streets-v12",
    "mapbox://styles/mapbox/outdoors-v12",
    "mapbox://styles/mapbox/satellite-v9",
    "mapbox://styles/mapbox/satellite-streets-v12",
    "mapbox://styles/mapbox/navigation-day-v1",
    "mapbox://styles/mapbox/navigation-night-v1",
]


def ensure_config() -> dict:
    """Initialize and return mutable session config."""
    if "app_config" not in st.session_state:
        st.session_state["app_config"] = DEFAULT_CONFIG.copy()
    return st.session_state["app_config"]


def get_config_value(key: str):
    """Get one config value with fallback to default."""
    cfg = ensure_config()
    return cfg.get(key, DEFAULT_CONFIG.get(key))


"""Session-scoped app configuration with project defaults."""

from math import gcd
import streamlit as st

MAPBOX_TOKEN = "pk.eyJ1Ijoibm90aG90dHJ5aGFyZCIsImEiOiJjbW54bWg5aWMwM2FxMnFyOHlkeTJ1ZG5pIn0.AiLtFFbOXt3MIouqO-cUag"  # public token for mapbox, good for commiting

DEFAULT_CONFIG = {
    "map_style": "mapbox://styles/mapbox/light-v11",
    "neighbor_level": 3,
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
    cfg = st.session_state["app_config"]
    for key, value in DEFAULT_CONFIG.items():
        cfg.setdefault(key, value)

    level = int(cfg.get("neighbor_level", 1))
    cfg["neighbor_level"] = _normalize_neighbor_level(level)

    # Backward-compatible init: if user has no explicit selection yet,
    # default to recommended primitive offsets for the configured level.
    if "neighbor_offsets" not in cfg:
        cfg["neighbor_offsets"] = recommended_offset_keys(cfg["neighbor_level"])
    return cfg


def get_config_value(key: str):
    """Get one config value with fallback to default."""
    cfg = ensure_config()
    return cfg.get(key, DEFAULT_CONFIG.get(key))


def _normalize_neighbor_level(level: int) -> int:
    level = max(1, int(level))
    return level if level % 2 == 1 else level + 1


def offset_key(dx: int, dy: int) -> str:
    return f"{dx},{dy}"


def parse_offset_key(key: str) -> tuple[int, int]:
    a, b = key.split(",")
    return int(a), int(b)


def base_offset_candidates(level: int) -> list[tuple[int, int]]:
    """Offsets for one octant: 0 <= dy <= dx <= level, excluding (0, 0)."""
    level = _normalize_neighbor_level(level)
    out = []
    for dx in range(1, level + 1):
        for dy in range(0, dx + 1):
            out.append((dx, dy))
    return out


def primitive_offset(dx: int, dy: int) -> bool:
    return gcd(dx, dy) == 1 if dy != 0 else dx == 1


def recommended_offset_keys(level: int) -> list[str]:
    return [
        offset_key(dx, dy)
        for dx, dy in base_offset_candidates(level)
        if primitive_offset(dx, dy)
    ]


def all_offset_keys(level: int) -> list[str]:
    return [offset_key(dx, dy) for dx, dy in base_offset_candidates(level)]


def expanded_directions(base_offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Expand first-octant offsets to all 8 directions."""
    out = set()
    for dx, dy in base_offsets:
        variants = {(dx, dy), (dy, dx)}
        for a, b in variants:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    out.add((sx * a, sy * b))
    return sorted(out, key=lambda v: (v[0] * v[0] + v[1] * v[1], abs(v[0]), abs(v[1]), v))


def get_neighbor_offsets() -> list[tuple[int, int]]:
    """Return expanded directions selected for current session config."""
    cfg = ensure_config()
    level = _normalize_neighbor_level(int(cfg.get("neighbor_level", 1)))
    cfg["neighbor_level"] = level

    valid_keys = set(all_offset_keys(level))
    selected_keys = cfg.get("neighbor_offsets")
    if not isinstance(selected_keys, list):
        selected_keys = recommended_offset_keys(level)
    selected_keys = [k for k in selected_keys if k in valid_keys]
    if not selected_keys:
        selected_keys = recommended_offset_keys(level)
    cfg["neighbor_offsets"] = selected_keys

    base = [parse_offset_key(k) for k in selected_keys]
    return expanded_directions(base)

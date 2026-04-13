"""Session configuration page."""

import altair as alt
import streamlit as st

from src.config import (
    DEFAULT_CONFIG,
    MAP_STYLE_OPTIONS,
    all_offset_keys,
    ensure_config,
    expanded_directions,
    parse_offset_key,
    recommended_offset_keys,
)

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

st.subheader("Межклеточные связи графа")

level_options = [1, 3, 5]
current_level = int(cfg.get("neighbor_level", DEFAULT_CONFIG["neighbor_level"]))
if current_level not in level_options:
    current_level = DEFAULT_CONFIG["neighbor_level"]

cfg["neighbor_level"] = st.select_slider(
    "Уровень связей (макс. шаг по сетке)",
    options=level_options,
    value=current_level,
)

all_keys = all_offset_keys(cfg["neighbor_level"])
recommended_keys = recommended_offset_keys(cfg["neighbor_level"])
selected_keys = cfg.get("neighbor_offsets", recommended_keys)
selected_keys = [k for k in selected_keys if k in set(all_keys)]
if not selected_keys:
    selected_keys = recommended_keys

col1, col2, col3 = st.columns(3)
if col1.button("Рекомендуемые"):
    selected_keys = recommended_keys.copy()
if col2.button("Все"):
    selected_keys = all_keys.copy()
if col3.button("Базовые 8-направлений"):
    selected_keys = [k for k in all_keys if k in {"1,0", "1,1"}]

def _fmt(key: str) -> str:
    dx, dy = parse_offset_key(key)
    return f"({dx}, {dy})"

selected_keys = st.multiselect(
    "Включённые смещения (для 1-го октанта; автоматически зеркалятся на 8 направлений)",
    options=all_keys,
    default=selected_keys,
    format_func=_fmt,
)
cfg["neighbor_offsets"] = selected_keys

status_order = [
    "Рекомендуемая и включена",
    "Добавлена вручную",
    "Рекомендуемая, но выключена",
    "Опциональная и выключена",
]
status_color = {
    "Рекомендуемая и включена": "#2ca02c",
    "Добавлена вручную": "#1f77b4",
    "Рекомендуемая, но выключена": "#ffbf00",
    "Опциональная и выключена": "#d9d9d9",
}

points = []
for key in all_keys:
    dx, dy = parse_offset_key(key)
    if key in selected_keys and key in recommended_keys:
        status = "Рекомендуемая и включена"
    elif key in selected_keys:
        status = "Добавлена вручную"
    elif key in recommended_keys:
        status = "Рекомендуемая, но выключена"
    else:
        status = "Опциональная и выключена"

    for dr, dc in expanded_directions([(dx, dy)]):
        points.append({"dx": dc, "dy": dr, "status": status})

if points:
    limit = cfg["neighbor_level"] + 0.5
    chart = (
        alt.Chart(alt.Data(values=points))
        .mark_circle(size=90)
        .encode(
            x=alt.X("dx:Q", title="Смещение по X (колонки)"),
            y=alt.Y("dy:Q", title="Смещение по Y (строки)"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=status_order,
                    range=[status_color[s] for s in status_order],
                ),
                legend=alt.Legend(title="Тип связи"),
            ),
        )
        .properties(height=420)
    )
    st.altair_chart(
        chart.configure_axis(grid=True).configure_view(strokeOpacity=0),
        use_container_width=True,
    )
    st.caption(
        f"Показаны направления вокруг центральной клетки в диапазоне ±{cfg['neighbor_level']}."
    )

if st.button("Сбросить настройки текущей сессии"):
    st.session_state["app_config"] = DEFAULT_CONFIG.copy()
    st.rerun()

st.caption("Стили mapbox:// могут требовать токен Mapbox в окружении.")

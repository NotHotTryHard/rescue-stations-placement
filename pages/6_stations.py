"""Manage active rescue stations for the current session."""

import pandas as pd
import streamlit as st

from src.session import (
    add_session_station,
    get_active_stations_raw,
    reset_active_stations,
    set_active_stations_raw,
)

st.set_page_config(page_title="Станции", layout="wide")
st.title("Станции")


def _records(data) -> list[dict]:
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return list(data)


stations = get_active_stations_raw()
edited = st.data_editor(
    pd.DataFrame(stations, columns=["id", "name", "lat", "lon", "speed_kmh"]),
    hide_index=True,
    num_rows="dynamic",
    width="stretch",
    column_config={
        "id": st.column_config.TextColumn("ID", required=True),
        "name": st.column_config.TextColumn("Название", required=True),
        "lat": st.column_config.NumberColumn("lat", format="%.6f", required=True),
        "lon": st.column_config.NumberColumn("lon", format="%.6f", required=True),
        "speed_kmh": st.column_config.NumberColumn(
            "Скорость, км/ч",
            min_value=1.0,
            step=1.0,
            format="%.1f",
            required=True,
        ),
    },
)

save_col, reset_col = st.columns([1, 1])
if save_col.button("Сохранить изменения", type="primary"):
    try:
        set_active_stations_raw(_records(edited))
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Не удалось сохранить станции: {exc}")
    else:
        st.success("Список станций обновлён.")
        st.rerun()

if reset_col.button("Сбросить к исходным станциям"):
    reset_active_stations()
    st.rerun()

st.subheader("Добавить станцию")
with st.form("add_station_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    name = col1.text_input("Название")
    station_id = col2.text_input("ID")
    col3, col4, col5 = st.columns(3)
    lat = col3.number_input("lat", value=60.0, format="%.6f")
    lon = col4.number_input("lon", value=29.85, format="%.6f")
    speed_kmh = col5.number_input("Скорость, км/ч", min_value=1.0, value=40.0, step=1.0)

    if st.form_submit_button("Добавить"):
        name = name.strip()
        station_id = station_id.strip() or name
        if not name:
            st.error("Укажи название станции.")
        elif not station_id:
            st.error("Укажи ID станции.")
        elif add_session_station(name, lat, lon, speed_kmh, station_id):
            st.success("Станция добавлена.")
            st.rerun()
        else:
            st.error("Станция с таким ID уже есть.")

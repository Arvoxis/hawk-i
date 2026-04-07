import streamlit as st
import folium
import requests
import time
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Hawk-I")
st.title("🦅 Hawk-I — Structural Inspection Dashboard")

# ── Query input ───────────────────────────────────────────────
col_q, col_btn = st.columns([4, 1])
with col_q:
    query = st.text_input("What should the drone look for?",
        placeholder="e.g. cracked concrete . rust stain . exposed rebar")
with col_btn:
    st.write("")
    if st.button("Send to Drone"):
        requests.post("http://localhost:8000/query", json={"query": query})
        st.success("Query sent!")

st.divider()

# ── Main layout ───────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Live Drone Feed")
    st.info("Video feed will appear here when drone connects")

with col2:
    st.subheader("Defect Map")

    # Initialize pins in session state
    if "pins" not in st.session_state:
        st.session_state.pins = []

    # Poll backend for latest detections
    try:
        resp = requests.get("http://localhost:8000/detections/latest", timeout=2)
        new_pins = resp.json()
        for pin in new_pins:
            if pin not in st.session_state.pins:
                st.session_state.pins.append(pin)
    except:
        pass

    # Build Folium map with all pins
    m = folium.Map(location=[12.97, 77.59], zoom_start=16)
    colour_map = {"L1": "green", "L2": "orange", "L3": "red"}

    for pin in st.session_state.pins:
        folium.CircleMarker(
            location=[pin["lat"], pin["lon"]],
            radius=10,
            color=colour_map.get(pin["severity"], "blue"),
            fill=True,
            fill_opacity=0.8,
            popup=f"{pin['class_name']} | {pin['severity']} | conf: {pin['confidence']}"
        ).add_to(m)

    st_folium(m, width=700, height=400, key="defect_map")

st.divider()

# ── Report panel ──────────────────────────────────────────────
st.subheader("Live Inspection Report")

if st.session_state.pins:
    for i, pin in enumerate(reversed(st.session_state.pins[-10:])):
        color = {"L1": "🟢", "L2": "🟡", "L3": "🔴"}.get(pin["severity"], "⚪")
        label = f"{pin['class_name'].upper()} — Severity: {pin['severity']} | Confidence: {round(pin['confidence']*100, 1)}% | GPS: {round(pin['lat'],5)}, {round(pin['lon'],5)}"
        st.markdown(f"{color} **{label}**")
"""Hawk-I — Aerial Infrastructure Inspector Dashboard."""
import asyncio
import base64
import json
import threading
import time

import requests
import folium
import streamlit as st
from datetime import datetime
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

from utils import (
    inject_css,
    render_stat_card,
    render_detection_card,
    render_report_panel,
    SEV_COLOR, SEV_LABEL, SEV_FOLIUM,
)

# ── WebSocket live-feed state ─────────────────────────────────────
import queue as _queue
_frame_queue: _queue.Queue = _queue.Queue(maxsize=1)
_ws_started = False


def _start_ws_listener():
    """Spawn a single daemon thread that keeps a WS connection open."""
    global _ws_started
    if _ws_started:
        return
    _ws_started = True

    def _run():
        async def _listen():
            import websockets  # local import to keep top-level clean
            while True:
                try:
                    async with websockets.connect(
                        "ws://localhost:8000/ws/dashboard",
                        ping_interval=10,
                        ping_timeout=5,
                    ) as ws:
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                                # Backend sends either a frame bundle (has "frame_jpeg" or
                                # "gps" key) or a legacy pin dict (has "lat"/"class" keys).
                                if "frame_jpeg" in msg or "gps" in msg:
                                    # Full frame bundle from backend
                                    payload = {
                                        "frame_jpeg":      msg.get("frame_jpeg"),
                                        "gps":             msg.get("gps", {}),
                                        "detections":      msg.get("detections", []),
                                        "last_frame_time": time.monotonic(),
                                    }
                                else:
                                    # Legacy pin-only message: promote to frame bundle shape
                                    payload = {
                                        "frame_jpeg":      None,
                                        "gps":             {
                                            "lat":   msg.get("lat", 0.0),
                                            "lon":   msg.get("lon", 0.0),
                                            "alt_m": msg.get("alt_m", 0.0),
                                        },
                                        "detections":      [msg],
                                        "last_frame_time": time.monotonic(),
                                    }
                                # Drop stale frame, put fresh one
                                try:
                                    _frame_queue.get_nowait()
                                except Exception:
                                    pass
                                try:
                                    _frame_queue.put_nowait(payload)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                except Exception:
                    await asyncio.sleep(1)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_listen())

    t = threading.Thread(target=_run, daemon=True, name="hawk-ws-listener")
    t.start()

# ── Page config (must be first Streamlit call) ───────────────────
st.set_page_config(layout="wide", page_title="Hawk-I | Aerial Inspector", page_icon="🦅")
inject_css()

BACKEND = "http://localhost:8000"

SEV_FILTER_MAP = {
    "All":      ["L3", "L2", "L1"],
    "Critical": ["L3"],
    "Moderate": ["L2"],
    "Low":      ["L1"],
}

# ── Data fetch ───────────────────────────────────────────────────
@st.cache_data(ttl=4)
def fetch_detections(limit: int = 100) -> list:
    try:
        r = requests.get(f"{BACKEND}/detections/latest", params={"limit": limit}, timeout=3)
        return r.json()
    except Exception:
        return []

all_detections = fetch_detections()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.35rem;font-weight:800;color:#F1F5F9;padding:.4rem 0'>🦅 Hawk-I</div>"
        "<div style='font-size:0.76rem;color:#475569;margin-bottom:.75rem'>"
        "Aerial Infrastructure Inspector</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Drone control
    st.markdown(
        "<div style='font-size:0.72rem;color:#64748B;text-transform:uppercase;"
        "letter-spacing:.06em;margin-bottom:6px'>Drone Control</div>",
        unsafe_allow_html=True,
    )
    query = st.text_input(
        "Query", placeholder="e.g. cracked concrete, rust stain",
        label_visibility="collapsed",
    )
    if st.button("🚁 Send to Drone", use_container_width=True):
        try:
            requests.post(f"{BACKEND}/query", json={"query": query}, timeout=2)
            st.success("Query sent!")
        except Exception:
            st.error("Backend offline")

    st.divider()

    # Class filter
    st.markdown(
        "<div style='font-size:0.72rem;color:#64748B;text-transform:uppercase;"
        "letter-spacing:.06em;margin-bottom:6px'>Filter by Class</div>",
        unsafe_allow_html=True,
    )
    classes = sorted({d["class_name"] for d in all_detections}) if all_detections else []
    selected_classes = st.multiselect(
        "Defect Class", classes, default=classes,
        label_visibility="collapsed", key="filter_class",
    )

    st.divider()

    # Live mode + live feed
    live_mode = st.toggle("🔴 Live Mode", value=True)
    live_feed = st.toggle("📹 Live Feed", value=False)
    if live_mode and not live_feed:
        st_autorefresh(interval=30000, key="hawk_autorefresh")
    st.caption("Auto-refreshes every 30 s when active (disabled during live feed)")

    if live_feed:
        _start_ws_listener()
    st.caption("Streams video from ws://localhost:8000/ws/dashboard")

    st.divider()

    # PDF report
    st.markdown(
        "<div style='font-size:0.72rem;color:#64748B;text-transform:uppercase;"
        "letter-spacing:.06em;margin-bottom:6px'>Reports</div>",
        unsafe_allow_html=True,
    )
    if st.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating…"):
            try:
                r = requests.get(f"{BACKEND}/api/report/pdf", timeout=60)
                if r.status_code == 200:
                    st.session_state["pdf_bytes"] = r.content
                else:
                    st.error(f"PDF failed ({r.status_code})")
            except Exception as exc:
                st.error(f"Backend error: {exc}")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            "📥 Download PDF",
            data=st.session_state["pdf_bytes"],
            file_name="hawki_inspection_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ── Header bar ───────────────────────────────────────────────────
now_str = datetime.now().strftime("%H:%M:%S  %d %b %Y")
st.markdown(
    f"""<div style="display:flex;justify-content:space-between;align-items:center;
        padding:.75rem 0 1rem;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:1.25rem">
        <div>
            <span style="font-size:1.5rem;font-weight:800;color:#F1F5F9">🦅 Hawk-I</span>
            <span style="font-size:0.85rem;color:#475569;margin-left:12px">
                Aerial Infrastructure Inspector</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:0.82rem;color:#94A3B8">
            <span class="live-dot" style="color:#10B981;font-size:1rem">●</span>
            <span style="color:#10B981;font-weight:600">LIVE</span>
            <span style="color:#334155">|</span>
            <span>{now_str}</span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Session info line ────────────────────────────────────────────
try:
    sess = requests.get(f"{BACKEND}/api/session", timeout=2).json()
    sess_id   = sess.get("session_id", "unknown")
    sess_dets = sess.get("detections", 0)
    sess_tbl  = sess.get("table", "—")
    st.caption(f"Session: `{sess_id}`  ·  table: `{sess_tbl}`  ·  {sess_dets} detections this run")
except Exception:
    pass

# ── Severity pill filter ─────────────────────────────────────────
sev_choice = st.radio(
    "Severity filter",
    options=list(SEV_FILTER_MAP.keys()),
    horizontal=True,
    label_visibility="collapsed",
    key="sev_filter",
)
selected_severities = SEV_FILTER_MAP[sev_choice]

# ── Apply filters ────────────────────────────────────────────────
pins = [
    d for d in all_detections
    if (not selected_classes or d["class_name"] in selected_classes)
    and d["severity"] in selected_severities
]

# ── Stat cards ───────────────────────────────────────────────────
total    = len(all_detections)
critical = sum(1 for d in all_detections if d["severity"] == "L3")
areas_m2 = [d["area_cm2"] / 10_000 for d in all_detections if d.get("area_cm2")]
area_total = sum(areas_m2)
sev_scores = {"L3": 3, "L2": 2, "L1": 1}
avg_sev = (
    sum(sev_scores.get(d["severity"], 0) for d in all_detections) / total
    if total else 0.0
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_stat_card(total, "Total Inspections", f"{total} logged"), unsafe_allow_html=True)
with c2:
    pct = f"{round(critical/total*100)}% of total" if total else "0%"
    st.markdown(render_stat_card(critical, "Critical Defects", pct, positive=False), unsafe_allow_html=True)
with c3:
    st.markdown(render_stat_card(f"{area_total:.2f} m²", "Area Scanned", f"{len(areas_m2)} measurements"), unsafe_allow_html=True)
with c4:
    st.markdown(render_stat_card(f"{avg_sev:.2f}/3", "Avg Severity Score",
                                 "Low" if avg_sev < 1.5 else ("Moderate" if avg_sev < 2.5 else "High"),
                                 positive=avg_sev < 2), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.25rem'></div>", unsafe_allow_html=True)

# ── Live video feed ───────────────────────────────────────────────
if live_feed:
    st.markdown(
        "<div style='font-size:0.78rem;color:#64748B;text-transform:uppercase;"
        "letter-spacing:.07em;margin-bottom:8px'>Live Video Feed</div>",
        unsafe_allow_html=True,
    )
    vid_col, gps_col = st.columns([3, 1])

    # Pull latest frame from queue into session_state so it persists across reruns
    try:
        _latest = _frame_queue.get_nowait()
        st.session_state["latest_frame"] = _latest
    except Exception:
        pass  # no new frame, use last known

    _snapshot  = st.session_state.get("latest_frame", {})
    frame_b64  = _snapshot.get("frame_jpeg")
    gps        = _snapshot.get("gps", {})
    ws_dets    = _snapshot.get("detections", [])
    last_t     = _snapshot.get("last_frame_time", 0)
    feed_fresh = last_t and (time.monotonic() - last_t) < 5

    with vid_col:
        if frame_b64 and feed_fresh:
            img_bytes = base64.b64decode(frame_b64)
            lat  = gps.get("lat", 0)
            lon  = gps.get("lon", 0)
            alt  = gps.get("alt_m", 0)
            caption = (
                f"GPS: {lat:.5f}, {lon:.5f}  |  Alt: {alt:.1f} m  "
                f"|  {len(ws_dets)} detection(s)"
            )
            st.image(img_bytes, caption=caption, use_container_width=True)
        else:
            st.markdown(
                '<div class="empty-state" style="padding:3rem 2rem">'
                '<span style="font-size:1.6rem">📡</span><br><br>'
                'Waiting for drone feed…<br>'
                '<span style="font-size:0.78rem">Connect the Jetson client to begin streaming</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    with gps_col:
        lat_v = f"{gps.get('lat', 0):.5f}" if feed_fresh else "—"
        lon_v = f"{gps.get('lon', 0):.5f}" if feed_fresh else "—"
        alt_v = f"{gps.get('alt_m', 0):.1f} m" if feed_fresh else "—"
        det_v = str(len(ws_dets)) if feed_fresh else "—"
        sev_counts = {}
        for d in ws_dets:
            sev_counts[d.get("severity", "?")] = sev_counts.get(d.get("severity", "?"), 0) + 1

        sev_rows = "".join(
            '<div style="margin-top:6px;font-size:0.78rem">'
            '<span style="color:{}">{}</span>: {}</div>'.format(
                SEV_COLOR.get(s, "#888"), SEV_LABEL.get(s, s), c
            )
            for s, c in sorted(sev_counts.items())
        )
        st.markdown(
            f"""<div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
                border-radius:12px;padding:1rem 1.1rem;height:100%">
              <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:.75rem">Telemetry</div>
              <div style="font-size:0.82rem;color:#94A3B8;line-height:2.1">
                <b style="color:#475569">LAT</b>&nbsp; {lat_v}<br>
                <b style="color:#475569">LON</b>&nbsp; {lon_v}<br>
                <b style="color:#475569">ALT</b>&nbsp; {alt_v}<br>
                <b style="color:#475569">DETS</b>&nbsp; {det_v}
              </div>
              {sev_rows}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:1.25rem'></div>", unsafe_allow_html=True)

# ── Main two-column layout ───────────────────────────────────────
map_col, feed_col = st.columns([13, 7])   # ~65% / 35%

# ---- Left: Folium map ------------------------------------------
with map_col:
    st.markdown(
        "<div style='font-size:0.78rem;color:#64748B;text-transform:uppercase;"
        "letter-spacing:.07em;margin-bottom:8px'>Defect Map</div>",
        unsafe_allow_html=True,
    )

    center = [pins[0]["lat"], pins[0]["lon"]] if pins else [12.97, 77.59]
    m = folium.Map(location=center, zoom_start=16, tiles="CartoDB dark_matter")

    for pin in pins:
        area_cm2   = pin.get("area_cm2")
        area_label = f"{area_cm2:.1f} cm²" if area_cm2 else "pending"
        conf_pct   = round(pin["confidence"] * 100, 1)
        sev        = pin["severity"]
        color_hex  = SEV_COLOR.get(sev, "#888")
        sev_lbl    = SEV_LABEL.get(sev, sev)

        report        = pin.get("llm_report") or ""
        report_lines  = [ln for ln in report.strip().split("\n") if ln.strip()]
        report_preview = "<br>".join(report_lines[:2]) if report_lines else ""

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:210px;font-size:13px;line-height:1.5">
            <b style="font-size:15px">{pin['class_name']}</b><br>
            <span style="display:inline-block;padding:1px 7px;border-radius:10px;
                background:{color_hex};color:#fff;font-size:11px;font-weight:700;margin:3px 0 5px">
                {sev_lbl.upper()}</span><br>
            <b>Confidence:</b> {conf_pct}%<br>
            <b>Area:</b> {area_label}<br>
            <b>GPS:</b> {round(pin['lat'],5)}, {round(pin['lon'],5)}
            {f'<hr style="margin:5px 0"><small style="color:#666">{report_preview}</small>'
             if report_preview else ''}
        </div>"""

        folium.CircleMarker(
            location=[pin["lat"], pin["lon"]],
            radius=11,
            color=SEV_FOLIUM.get(sev, "blue"),
            fill=True,
            fill_color=SEV_FOLIUM.get(sev, "blue"),
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(popup_html, max_width=290),
            tooltip=folium.Tooltip(
                f"<b>{pin['class_name']}</b> | {sev_lbl} | {conf_pct}%",
                sticky=True,
            ),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=470, key="defect_map")

# ---- Right: Detection feed -------------------------------------
with feed_col:
    st.markdown(
        "<div style='font-size:0.78rem;color:#64748B;text-transform:uppercase;"
        "letter-spacing:.07em;margin-bottom:8px'>Live Detection Feed</div>",
        unsafe_allow_html=True,
    )
    recent = pins[:12]

    if not recent:
        st.markdown(
            '<div class="empty-state">No detections yet.<br>'
            '<span style="font-size:0.8rem">Start the drone to begin inspection.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        for d in recent:
            st.markdown(render_detection_card(d), unsafe_allow_html=True)

# ── Inspection reports ───────────────────────────────────────────
st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.78rem;color:#64748B;text-transform:uppercase;"
    "letter-spacing:.07em;margin-bottom:10px'>Inspection Reports</div>",
    unsafe_allow_html=True,
)

report_pins = [p for p in pins[:10] if p.get("id")]

if not report_pins:
    st.markdown(
        '<div class="empty-state">No reports available yet.</div>',
        unsafe_allow_html=True,
    )
else:
    for det in report_pins:
        sev   = det["severity"]
        label = SEV_LABEL.get(sev, sev).upper()
        ts    = str(det.get("detected_at", ""))[:19].replace("T", " ")
        title = (
            f"#{det['id']}  ·  {det['class_name']}  ·  "
            f"{label}  ·  {round(det['lat'],4)}, {round(det['lon'],4)}  ·  {ts}"
        )

        with st.expander(title):
            # Fetch report if not already cached on the object
            if not det.get("llm_report"):
                try:
                    r = requests.get(f"{BACKEND}/api/report/{det['id']}", timeout=3)
                    if r.status_code == 200:
                        det["llm_report"] = r.json().get("report")
                except Exception:
                    pass

            render_report_panel(det)

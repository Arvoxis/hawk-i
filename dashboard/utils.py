"""Shared helpers and CSS for the Hawk-I dashboard."""
import streamlit as st

# ── Severity mappings ────────────────────────────────────────────
SEV_COLOR  = {"L3": "#EF4444", "L2": "#F59E0B", "L1": "#10B981"}
SEV_LABEL  = {"L3": "Critical", "L2": "Moderate", "L1": "Low"}
SEV_BADGE  = {"L3": "badge-critical", "L2": "badge-moderate", "L1": "badge-low"}
SEV_FOLIUM = {"L3": "red", "L2": "orange", "L1": "green"}


def inject_css():
    """Inject all custom CSS — called once at app startup."""
    st.markdown("""<style>
    /* ── Fonts & global reset ─────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        background-color: #0A0E1A !important;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #E2E8F0 !important;
    }
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: #1E293B !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
        background: #1E293B !important;
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button {
        background: #3B82F6 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button:hover { background: #2563EB !important; }
    .stDownloadButton > button {
        background: transparent !important;
        color: #3B82F6 !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover { background: rgba(59,130,246,0.1) !important; }

    /* ── Inputs ───────────────────────────────────────────── */
    .stTextInput input {
        background: #1E293B !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }
    .stMultiSelect [data-baseweb="tag"] { background: #3B82F6 !important; }

    /* ── Toggles ──────────────────────────────────────────── */
    [data-testid="stToggle"] label { color: #94A3B8 !important; }

    /* ── Radio → pill buttons ─────────────────────────────── */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        gap: 8px !important;
        flex-wrap: wrap !important;
    }
    div[data-testid="stRadio"] label {
        background: #1E293B !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 20px !important;
        padding: 5px 18px !important;
        cursor: pointer !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        transition: all 0.15s !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #3B82F6 !important;
        border-color: #3B82F6 !important;
        color: #fff !important;
    }
    div[data-testid="stRadio"] input {
        opacity: 0;
        position: absolute;
        width: 0;
        height: 0;
    }

    /* ── Expanders ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #111827 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 8px !important;
        color: #CBD5E1 !important;
        font-size: 0.88rem !important;
    }
    .streamlit-expanderContent {
        background: #111827 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* ── Misc ─────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.07) !important; }
    iframe { border-radius: 12px !important; }
    .stAlert { border-radius: 8px !important; }
    .stSpinner > div { border-top-color: #3B82F6 !important; }

    /* ── LIVE pulse keyframe ──────────────────────────────── */
    @keyframes livePulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.2; }
    }
    .live-dot { animation: livePulse 1.5s ease-in-out infinite; }

    /* ── Stat cards ───────────────────────────────────────── */
    .stat-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F1F5F9;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 5px;
    }
    .stat-delta { font-size: 0.75rem; margin-top: 8px; font-weight: 500; }

    /* ── Detection feed ───────────────────────────────────── */
    .feed-container {
        max-height: 470px;
        overflow-y: auto;
        padding-right: 2px;
    }
    .feed-container::-webkit-scrollbar { width: 3px; }
    .feed-container::-webkit-scrollbar-track { background: transparent; }
    .feed-container::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.12);
        border-radius: 2px;
    }
    .detection-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-left: 3px solid transparent;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: border-left-color 0.2s;
    }
    .detection-card:hover { border-left-color: #3B82F6 !important; }

    /* ── Severity badges ──────────────────────────────────── */
    .severity-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-critical {
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        border: 1px solid rgba(239,68,68,0.35);
    }
    .badge-moderate {
        background: rgba(245,158,11,0.15);
        color: #F59E0B;
        border: 1px solid rgba(245,158,11,0.35);
    }
    .badge-low {
        background: rgba(16,185,129,0.15);
        color: #10B981;
        border: 1px solid rgba(16,185,129,0.35);
    }

    /* ── Report panel ─────────────────────────────────────── */
    .report-panel {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
    }

    /* ── Empty state ──────────────────────────────────────── */
    .empty-state {
        background: #111827;
        border: 1px dashed rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        color: #475569;
        font-size: 0.9rem;
    }
    </style>""", unsafe_allow_html=True)


def render_stat_card(value, label, delta=None, positive=True):
    """Return HTML for a single stat card."""
    delta_color = "#10B981" if positive else "#EF4444"
    arrow = "▲" if positive else "▼"
    delta_html = (
        f'<div class="stat-delta" style="color:{delta_color}">{arrow} {delta}</div>'
        if delta else
        '<div class="stat-delta" style="color:#334155">—</div>'
    )
    return f"""<div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        {delta_html}
    </div>"""


def _esc(value: str) -> str:
    """Minimal HTML-escape for dynamic values inserted into card markup."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_detection_card(det):
    """Return HTML for a single detection feed card."""
    sev      = det["severity"]
    badge    = SEV_BADGE.get(sev, "badge-low")
    label    = _esc(SEV_LABEL.get(sev, sev))
    cls_name = _esc(det["class_name"])
    conf     = round(det["confidence"] * 100, 1)
    area     = det.get("area_cm2")
    area_str = f"{area:.1f} cm\u00b2" if area else "\u2014"
    ts       = _esc(str(det.get("detected_at", ""))[:19].replace("T", " "))
    lat      = round(det["lat"], 4)
    lon      = round(det["lon"], 4)

    report  = det.get("llm_report") or ""
    summary = next((ln.strip() for ln in report.split("\n") if ln.strip()), "")
    summary_html = (
        '<div style="font-size:11px;color:#475569;margin-top:5px;'
        'font-style:italic;line-height:1.4">{}\u2026</div>'.format(_esc(summary[:115]))
    ) if summary else ""

    return (
        '<div class="detection-card">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        '<span style="font-size:14px;font-weight:700;color:#F1F5F9">{cls}</span>'
        '<span class="severity-badge {badge}">{label}</span>'
        "</div>"
        '<div style="font-size:12px;color:#64748B;line-height:1.9">'
        "\U0001f3af {conf}% &nbsp;\u00b7&nbsp; \U0001f4d0 {area} &nbsp;\u00b7&nbsp; \U0001f4cd {lat}, {lon}"
        "</div>"
        "{summary}"
        '<div style="font-size:11px;color:#334155;margin-top:4px">\U0001f550 {ts}</div>'
        "</div>"
    ).format(
        cls=cls_name, badge=badge, label=label,
        conf=conf, area=area_str, lat=lat, lon=lon,
        summary=summary_html, ts=ts,
    )


def render_report_panel(det):
    """Render a styled report panel inside a Streamlit expander (uses st calls)."""
    sev   = det["severity"]
    badge = SEV_BADGE.get(sev, "badge-low")
    label = SEV_LABEL.get(sev, sev)
    conf  = round(det["confidence"] * 100, 1)
    area  = det.get("area_cm2")
    ts    = str(det.get("detected_at", ""))[:19].replace("T", " ")
    alt   = det.get("altitude_m")

    def _cell(k, v):
        return (
            f'<div style="margin-bottom:8px">'
            f'<div style="font-size:10px;color:#475569;text-transform:uppercase;'
            f'letter-spacing:.06em;margin-bottom:2px">{k}</div>'
            f'<div style="font-size:13px;font-weight:600;color:#CBD5E1">{v}</div>'
            f'</div>'
        )

    altitude_cell = _cell("Altitude", f"{alt} m") if alt else ""
    details_html = f"""<div class="report-panel" style="margin-bottom:1rem">
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px 16px">
            {_cell("Class", det['class_name'])}
            {_cell("Severity", f'<span class="severity-badge {badge}">{label}</span>')}
            {_cell("Confidence", f'{conf}%')}
            {_cell("Area", f'{area:.1f} cm²' if area else 'Pending SAM2')}
            {_cell("GPS", f'{round(det["lat"],5)}, {round(det["lon"],5)}')}
            {_cell("Timestamp", ts)}
            {altitude_cell}
        </div>
    </div>"""

    st.markdown(details_html, unsafe_allow_html=True)

    report_text = det.get("llm_report")
    if report_text:
        # Escape the report text and render it inside the styled panel
        report_html = "<br>".join(_esc(line) for line in report_text.strip().splitlines())
        st.markdown(
            '<div class="report-panel">'
            '<div style="font-size:10px;color:#3B82F6;text-transform:uppercase;'
            'letter-spacing:.08em;font-weight:700;margin-bottom:10px">Gemma-3 Inspection Report</div>'
            f'<div style="font-size:13px;color:#CBD5E1;line-height:1.7;white-space:pre-wrap">'
            f'{report_html}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="empty-state" style="padding:1rem">'
            'Report not yet generated — SAM2 / LLM pipeline may still be running.'
            '</div>',
            unsafe_allow_html=True,
        )

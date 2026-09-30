"""
Hawk-I PDF Report Generator — Enhanced Pipeline
================================================
Generates a professional, multi-section A4 inspection report with:
  • Mission cover metadata table
  • Site health gauge bar
  • Per-class defect distribution chart
  • Confidence band breakdown table
  • Defect detail cards (thumbnail + LLM report)
  • Full detection data table
  • Detection map (coordinate-plane scatter)
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Optional

import cv2
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, Circle, String, Line, Polygon
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Fonts
#
# ReportLab's built-in Helvetica is Latin-1 only, so the rupee sign, the
# warning triangle and the filled square all rendered as tofu boxes -- in a
# report whose entire purpose is quoting Indian repair costs.  Vera.ttf ships
# with ReportLab but predates the 2010 rupee sign, so it does not help.
#
# DejaVuSans covers everything.  It is not a declared dependency, so we look
# for it in the usual places (including matplotlib's bundle, which is present
# on most scientific installs) and fall back to sanitising the text when no
# Unicode font can be found.  The report is then still correct, just plainer.
# ─────────────────────────────────────────────────────────────────

def _font_candidates() -> list[str]:
    paths: list[str] = []

    override = os.getenv("HAWKI_PDF_FONT")
    if override:
        paths.append(override)

    try:
        import matplotlib
        mpl_ttf = os.path.join(
            os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"
        )
        paths.append(os.path.join(mpl_ttf, "DejaVuSans.ttf"))
    except Exception:
        pass

    paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
    ]
    return paths


def _bold_variant(path: str) -> Optional[str]:
    """Locate the bold face sitting next to a regular one."""
    for suffix in ("-Bold", "Bd", "bd", "-bold"):
        stem, ext = os.path.splitext(path)
        candidate = f"{stem}{suffix}{ext}"
        if os.path.exists(candidate):
            return candidate
    return None


def _register_fonts() -> tuple[str, str, bool]:
    """Return (regular, bold, unicode_ok)."""
    for path in _font_candidates():
        if not path or not os.path.exists(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont("HawkiSans", path))
            bold_path = _bold_variant(path)
            if bold_path:
                pdfmetrics.registerFont(TTFont("HawkiSans-Bold", bold_path))
                bold_name = "HawkiSans-Bold"
            else:
                bold_name = "HawkiSans"
            pdfmetrics.registerFontFamily(
                "HawkiSans", normal="HawkiSans", bold=bold_name,
                italic="HawkiSans", boldItalic=bold_name,
            )
            logger.info("PDF fonts: using Unicode font %s", os.path.basename(path))
            return "HawkiSans", bold_name, True
        except Exception as exc:
            logger.debug("PDF fonts: %s unusable (%s)", path, exc)

    logger.warning(
        "PDF fonts: no Unicode TTF found -- falling back to Helvetica and "
        "transliterating symbols (INR for the rupee sign, etc). "
        "Install matplotlib or set HAWKI_PDF_FONT to a DejaVuSans.ttf for "
        "full symbol support."
    )
    return FONT, FONT_BOLD, False


FONT, FONT_BOLD, UNICODE_OK = _register_fonts()

# Characters Helvetica cannot draw, and what to write instead.
_TRANSLITERATIONS = {
    "\u20b9": "INR ",   # rupee sign
    "\u2011": "-",      # non-breaking hyphen
    "\u26a0": "!",      # warning triangle
    "\u25a0": "*",      # filled square
    "\u2265": ">=",
    "\u2264": "<=",
    "\u2013": "-",      # en dash
    "\u2014": "-",      # em dash
    "\u2026": "...",
    "\u00b7": "-",      # middle dot
    "\u00b2": "2",      # superscript two (cm2)
}


def _t(text) -> str:
    """Make a string safe for the active font.

    A no-op when a Unicode font is registered; otherwise transliterates the
    characters Helvetica would render as empty boxes.
    """
    text = "" if text is None else str(text)
    if UNICODE_OK:
        return text
    for bad, good in _TRANSLITERATIONS.items():
        text = text.replace(bad, good)
    return text


def _esc(text) -> str:
    """Escape XML metacharacters before they reach a Paragraph.

    Defect class names come from model labels and operator queries, so an
    ampersand in one would otherwise abort rendering of the whole report.
    """
    return (
        _t(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

# ── Palette ────────────────────────────────────────────────────
SEV_BG    = {"L3": colors.HexColor("#fde8ea"), "L2": colors.HexColor("#fff3e0"), "L1": colors.HexColor("#e8f5e9")}
SEV_FG    = {"L3": colors.HexColor("#c0392b"), "L2": colors.HexColor("#d35400"), "L1": colors.HexColor("#1e8449")}
SEV_LABEL = {"L3": "HIGH",                    "L2": "MEDIUM",                   "L1": "LOW"}
SEV_DRAW  = {
    "L3": colors.HexColor("#e74c3c"),
    "L2": colors.HexColor("#e67e22"),
    "L1": colors.HexColor("#27ae60"),
}

NAVY  = colors.HexColor("#1a1a2e")
CYAN  = colors.HexColor("#00b4d8")
GREY  = colors.HexColor("#6c757d")
LIGHT = colors.HexColor("#f8f9fa")
LINE  = colors.HexColor("#dee2e6")
WHITE = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _area_str(det: dict) -> str:
    """Smart area label: real SAM / bbox-estimate / pending."""
    area     = det.get("area_cm2") or 0.0
    sam_score = det.get("sam_score", 0.0) or 0.0
    if area > 0:
        if sam_score < 0:  # sentinel: bbox-estimated
            return _t(f"~{area:.1f} cm\u00b2 (est.)")
        return _t(f"{area:.1f} cm\u00b2")
    return _t("\u2014")  # em-dash: truly no data


def _draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, 1.1 * cm,
                      _t("Generated by Hawk-I \u2014 Aerial Infrastructure Inspector"))
    canvas.drawRightString(PAGE_W - MARGIN, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
    canvas.restoreState()


def _styles() -> dict:
    base = getSampleStyleSheet()

    def make(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    for style in base.byName.values():
        if getattr(style, "fontName", "").startswith("Helvetica"):
            style.fontName = FONT_BOLD if "Bold" in style.fontName else FONT

    return {
        # 19pt with explicit leading: at 22pt the title wrapped onto a second
        # line whose default leading was too tight, so it printed on top of
        # the subtitle.
        "title":    make("ReportTitle",  fontSize=19, leading=23, fontName=FONT_BOLD,
                         textColor=NAVY, spaceAfter=4, alignment=TA_LEFT),
        "subtitle": make("ReportSub",    fontSize=10, fontName=FONT,
                         textColor=GREY, spaceAfter=3),
        "section":  make("Section",      fontSize=12, fontName=FONT_BOLD,
                         textColor=NAVY, spaceBefore=14, spaceAfter=6),
        "body":     make("Body",         fontSize=10, leading=14),
        "small":    make("Small",        fontSize=8,  textColor=GREY, leading=11),
        "report":   make("ReportText",   fontSize=9,  leading=13, leftIndent=10,
                         textColor=colors.HexColor("#333333")),
        "cell":     make("Cell",         fontSize=8,  leading=11),
        "badge":    make("Badge",        fontSize=8,  fontName=FONT_BOLD,
                         alignment=TA_CENTER),
    }


# ─────────────────────────────────────────────────────────────────
# Thumbnail loader
# ─────────────────────────────────────────────────────────────────

def _load_thumbnail(
    image_path: str | None,
    raw_box_json: str | None,
    thumb_w: float = 120,
    thumb_h: float = 80,
) -> Optional[RLImage]:
    if not image_path or not os.path.isfile(image_path):
        return None
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None

    if raw_box_json:
        try:
            box = json.loads(raw_box_json)
            x1, y1, x2, y2 = [int(v) for v in box]
            pad = 20
            h, w = bgr.shape[:2]
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
            if x2 > x1 and y2 > y1:
                bgr = bgr[y1:y2, x1:x2]
        except Exception:
            pass

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return RLImage(BytesIO(buf.tobytes()), width=thumb_w, height=thumb_h)


# ─────────────────────────────────────────────────────────────────
# Site health gauge bar
# ─────────────────────────────────────────────────────────────────

def _build_health_gauge(score: int, draw_w: float = 480, draw_h: float = 48) -> Drawing:
    """Horizontal health bar: green→yellow→red gradient segments + score label."""
    d = Drawing(draw_w, draw_h)

    bar_x = 10
    bar_y = 14
    bar_w = draw_w - 80
    bar_h = 18

    # Background track
    d.add(Rect(bar_x, bar_y, bar_w, bar_h,
               fillColor=colors.HexColor("#e9ecef"), strokeColor=None))

    # Filled portion
    fill_w = max(4, bar_w * score / 100)
    if score > 60:
        fill_col = colors.HexColor("#27ae60")
    elif score > 30:
        fill_col = colors.HexColor("#f39c12")
    else:
        fill_col = colors.HexColor("#e74c3c")

    d.add(Rect(bar_x, bar_y, fill_w, bar_h,
               fillColor=fill_col, strokeColor=None))

    # Border
    d.add(Rect(bar_x, bar_y, bar_w, bar_h,
               fillColor=None, strokeColor=colors.HexColor("#adb5bd"), strokeWidth=0.5))

    # Score label
    d.add(String(bar_x + bar_w + 8, bar_y + 4,
                 f"{score}/100",
                 fontName=FONT_BOLD, fontSize=11, fillColor=fill_col))

    # Label
    d.add(String(bar_x, bar_y - 8, "SITE HEALTH SCORE",
                 fontName=FONT_BOLD, fontSize=7, fillColor=GREY))

    return d


# ─────────────────────────────────────────────────────────────────
# Per-class horizontal bar chart
# ─────────────────────────────────────────────────────────────────

def _build_class_chart(detections: list, draw_w: float = 480) -> Drawing:
    """Horizontal bar chart showing detection count per defect class."""
    class_counts = Counter(d.get("class_name", "unknown") for d in detections)
    if not class_counts:
        return Drawing(draw_w, 20)

    # Sort by count descending, cap at 10 classes
    items = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
    n = len(items)
    bar_h = 14
    gap   = 7
    label_w = 150          # was 130 -- "Exposed_reinforcement" was being cut
    count_w = 28           # reserved gutter so the count never sits outside
    row_h = bar_h + gap
    draw_h = n * row_h + 30

    d = Drawing(draw_w, draw_h)
    # 15% headroom: at max_count the bar previously filled the frame edge to
    # edge and its value label was drawn past the right border.
    max_count = max(v for _, v in items) * 1.15
    chart_w   = draw_w - label_w - count_w - 10

    d.add(String(0, draw_h - 12, "Detections by Defect Class",
                 fontName=FONT_BOLD, fontSize=9, fillColor=NAVY))

    for i, (cls, cnt) in enumerate(items):
        y = draw_h - 28 - i * row_h
        bar_px = max(4, chart_w * cnt / max_count)

        d.add(Rect(label_w, y, bar_px, bar_h,
                   fillColor=CYAN, strokeColor=None))
        d.add(Rect(label_w, y, chart_w, bar_h,
                   fillColor=None,
                   strokeColor=colors.HexColor("#dee2e6"), strokeWidth=0.4))

        # Class label, truncated to what actually fits the reserved width
        lbl = _t(cls)
        while lbl and pdfmetrics.stringWidth(lbl, FONT, 7.5) > label_w - 8:
            lbl = lbl[:-1]
        if lbl != _t(cls):
            lbl = lbl[:-1] + "\u2026" if UNICODE_OK else lbl[:-3] + "..."
        d.add(String(0, y + 3, lbl,
                     fontName=FONT, fontSize=7.5, fillColor=colors.HexColor("#333")))

        # Count label, in its own gutter to the right of the frame
        d.add(String(label_w + chart_w + 6, y + 3, str(cnt),
                     fontName=FONT_BOLD, fontSize=7.5, fillColor=NAVY))

    return d


# ─────────────────────────────────────────────────────────────────
# Confidence band breakdown table
# ─────────────────────────────────────────────────────────────────

def _confidence_table(detections: list, S: dict) -> Table:
    """Three-band confidence breakdown: High / Medium / Low."""
    high   = [d for d in detections if (d.get("confidence") or 0) >= 0.85]
    medium = [d for d in detections if 0.65 <= (d.get("confidence") or 0) < 0.85]
    low    = [d for d in detections if (d.get("confidence") or 0) < 0.65]

    rows = [
        ["Confidence Band", "Range",        "Count", "% of Total"],
        ["High",            "\u226585%",     str(len(high)),   f"{100*len(high)/max(1,len(detections)):.1f}%"],
        ["Medium",          "65\u201384%",   str(len(medium)), f"{100*len(medium)/max(1,len(detections)):.1f}%"],
        ["Low",             "<65%",         str(len(low)),    f"{100*len(low)/max(1,len(detections)):.1f}%"],
        ["Total",           "",             str(len(detections)), "100%"],
    ]
    t = Table(rows, colWidths=[5 * cm, 3 * cm, 3 * cm, 4.5 * cm])
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("FONTNAME",      (0, 1), (-1, -1), FONT),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -2), [LIGHT, WHITE]),
        ("BACKGROUND",    (0, -1), (-1, -1), colors.HexColor("#e9ecef")),
        ("FONTNAME",      (0, -1), (-1, -1), FONT_BOLD),
        ("GRID",          (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",         (2, 0), (3, -1), "CENTER"),
        # Color the High-confidence count cell
        ("TEXTCOLOR",     (2, 1), (2, 1), colors.HexColor("#1e8449")),
        ("FONTNAME",      (2, 1), (2, 1), FONT_BOLD),
    ])
    t.setStyle(ts)
    return t


# ─────────────────────────────────────────────────────────────────
# Executive summary key-value table
# ─────────────────────────────────────────────────────────────────

def _summary_table(detections: list, S: dict) -> Table:
    by_sev = Counter(d.get("severity", "L1") for d in detections)
    areas  = [d["area_cm2"] for d in detections if (d.get("area_cm2") or 0) > 0
              and (d.get("sam_score") or 0) >= 0]   # exclude bbox-estimates from avg
    total_area = sum(areas)
    avg_area   = total_area / len(areas) if areas else 0.0
    class_counts = Counter(d.get("class_name", "?") for d in detections)
    most_common  = class_counts.most_common(1)[0][0] if class_counts else "—"

    # Compute % of detections that have real area
    real_area_pct = f"{100*len(areas)/max(1,len(detections)):.0f}%"

    rows = [
        ["Metric", "Value"],
        ["Total Detections",         str(len(detections))],
        ["Critical  (L3 / HIGH)",    str(by_sev.get("L3", 0))],
        ["Medium    (L2 / MEDIUM)",  str(by_sev.get("L2", 0))],
        ["Low       (L1 / LOW)",     str(by_sev.get("L1", 0))],
        ["Total Defect Area (SAM)",  f"{total_area:.1f} cm\u00b2"],
        ["Average Defect Area",      f"{avg_area:.1f} cm\u00b2  ({real_area_pct} SAM-measured)"],
        ["Most Common Defect Type",  most_common],
    ]

    t = Table(rows, colWidths=[9.5 * cm, 7.5 * cm])
    style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("FONTNAME",      (0, 1), (0, -1), FONT_BOLD),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
        ("BACKGROUND",    (1, 2), (1, 2),
         SEV_BG["L3"] if by_sev.get("L3", 0) > 0 else LIGHT),
        ("TEXTCOLOR",     (1, 2), (1, 2),
         SEV_FG["L3"] if by_sev.get("L3", 0) > 0 else colors.black),
        ("FONTNAME",      (1, 2), (1, 2), FONT_BOLD),
    ])
    t.setStyle(style)
    return t


# ─────────────────────────────────────────────────────────────────
# Full detection data table
# ─────────────────────────────────────────────────────────────────

def _detection_table(detections: list) -> Table:
    header = ["ID", "Defect Type", "Severity", "Area", "Conf %", "GPS Coordinates", "Timestamp"]
    rows = [header]
    sev_rows = []

    for det in detections:
        sev = det.get("severity", "L1")
        rows.append([
            str(det.get("id", "—")),
            det.get("class_name", "?"),
            f"{sev} / {SEV_LABEL.get(sev, sev)}",
            _area_str(det),
            f"{round((det.get('confidence') or 0) * 100, 1)}%",
            f"{round(det.get('lat') or 0, 4)}, {round(det.get('lon') or 0, 4)}",
            str(det.get("detected_at", ""))[:16],
        ])
        sev_rows.append((len(rows) - 1, sev))

    col_w = [1.2*cm, 3.6*cm, 2.6*cm, 2.2*cm, 1.6*cm, 3.5*cm, 2.3*cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("FONTNAME",      (0, 1), (-1, -1), FONT),
        ("FONTSIZE",      (0, 1), (-1, -1), 7.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.3, LINE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ("ALIGN",         (3, 0), (4, -1), "CENTER"),
    ])
    for row_idx, sev in sev_rows:
        ts.add("BACKGROUND", (2, row_idx), (2, row_idx), SEV_BG.get(sev, WHITE))
        ts.add("TEXTCOLOR",  (2, row_idx), (2, row_idx), SEV_FG.get(sev, colors.black))
        ts.add("FONTNAME",   (2, row_idx), (2, row_idx), FONT_BOLD)
    t.setStyle(ts)
    return t


# ─────────────────────────────────────────────────────────────────
# Defect detail cards (one per detection)
# ─────────────────────────────────────────────────────────────────

def _defect_detail_cards(detections: list, S: dict) -> list:
    flowables = []
    flowables.append(Paragraph("Defect Details", S["section"]))
    flowables.append(Paragraph(
        f"All {len(detections)} detection(s) sorted by severity (highest first), "
        "then by area descending.",
        S["body"],
    ))
    flowables.append(Spacer(1, 8))

    sev_order = {"L3": 0, "L2": 1, "L1": 2}
    sorted_dets = sorted(
        detections,
        key=lambda d: (sev_order.get(d.get("severity", "L1"), 2),
                       -(d.get("area_cm2") or 0))
    )

    for det in sorted_dets:
        sev  = det.get("severity", "L1")
        conf = round((det.get("confidence") or 0) * 100, 1)
        ts   = str(det.get("detected_at", ""))[:19]
        lat  = round(det.get("lat") or 0, 5)
        lon  = round(det.get("lon") or 0, 5)
        area = _area_str(det)

        dino_text = _t(" \u26a0 DINOv2 FLAGGED") if det.get("dinov2_flagged") else ""

        sim_text = ""
        try:
            ids = json.loads(det.get("similar_ids") or "[]")
            if ids:
                sim_text = f"  Similar past defects: IDs {ids}"
        except Exception:
            pass

        thumb = _load_thumbnail(
            det.get("image_path"),
            det.get("raw_box_json"),
            thumb_w=90, thumb_h=60,
        )

        info_lines = [
            "<b>#{} \u2014 {}</b>  [{}]{}".format(
                det.get("id", "?"), _esc(det.get("class_name", "?")),
                SEV_LABEL.get(sev, sev), dino_text),
            f"Area: {area}  |  Conf: {conf}%  |  {ts}",
            f"GPS: {lat}, {lon}  |  Alt: {det.get('altitude_m', 0):.1f} m",
            f"Source: {det.get('source_model', '—')}{sim_text}",
        ]

        llm_report = (det.get("llm_report") or "").strip()
        if llm_report:
            if llm_report.startswith("{"):
                try:
                    parsed  = json.loads(llm_report)
                    action  = parsed.get("recommended_action", "")
                    urgency = parsed.get("urgency_days", "")
                    desc    = parsed.get("description", "")
                    cost    = parsed.get("estimated_cost_inr")
                    cost_s  = _t(f"\u20b9{cost:,}") if isinstance(cost, int) else ""
                    if desc:
                        info_lines.append(f"Assessment: {_esc(desc)}")
                    if action:
                        info_lines.append(
                            f"Action: {_esc(action)}" + (f"  ({_esc(urgency)}d)" if urgency else "")
                        )
                    if cost_s:
                        info_lines.append(f"Est. cost: {cost_s}")
                except Exception:
                    info_lines.append(f"Report: {llm_report[:100].replace(chr(10),' ')}…")
            else:
                info_lines.append(f"Report: {llm_report[:120].replace(chr(10),' ')}…")

        # One Paragraph per ROW.  This was previously a single row of N
        # columns, which laid the four info lines out side by side and pushed
        # everything past "Area:" off the right edge of the page.
        info_col = [[Paragraph(line, S["cell"])] for line in info_lines]

        if thumb:
            row      = [[thumb, Table(info_col, colWidths=[11 * cm])]]
            col_widths = [3.2 * cm, 11 * cm]
        else:
            row      = [[Table(info_col, colWidths=[14 * cm])]]
            col_widths = [14.2 * cm]

        card = Table(row, colWidths=col_widths)
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), SEV_BG.get(sev, LIGHT)),
            ("LINEABOVE",     (0, 0), (-1, 0), 2.5, SEV_FG.get(sev, GREY)),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        flowables.append(KeepTogether([card, Spacer(1, 6)]))

    return flowables


# ─────────────────────────────────────────────────────────────────
# Critical LLM report section
# ─────────────────────────────────────────────────────────────────

def _critical_details(critical: list, S: dict) -> list:
    flowables = [Paragraph("Critical Defect Analysis", S["section"])]
    flowables.append(Paragraph(
        f"The following {len(critical)} detection(s) were measured at L3/HIGH "
        "severity from their segmented area. Each carries a generated "
        "inspection assessment; where that assessment grades the defect "
        "differently from the measurement, both are shown.",
        S["body"],
    ))
    flowables.append(Spacer(1, 8))

    for det in critical:
        area_s   = _area_str(det)
        conf_pct = round((det.get("confidence") or 0) * 100, 1)
        ts       = str(det.get("detected_at", ""))[:19]

        header_row = [[
            Paragraph("<b>#{} \u2014 {}</b>".format(
                det.get("id", "?"), _esc(det.get("class_name", "?"))), S["body"]),
            Paragraph(
                f"GPS: {round(det.get('lat') or 0, 5)}, {round(det.get('lon') or 0, 5)} &nbsp;|&nbsp; "
                f"Area: {area_s} &nbsp;|&nbsp; Confidence: {conf_pct}% &nbsp;|&nbsp; {ts}",
                S["small"],
            ),
        ]]
        card_header = Table(header_row, colWidths=[7 * cm, 10 * cm])
        card_header.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), SEV_BG["L3"]),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEABOVE",     (0, 0), (-1, 0), 2.5, SEV_FG["L3"]),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))

        report_paras = [card_header]
        raw = (det.get("llm_report") or "").strip()
        if raw.startswith("{"):
            try:
                r = json.loads(raw)
                em  = "—"
                cost = r.get("estimated_cost_inr")
                cost_str = (_t("\u20b9{:,}".format(cost)) if isinstance(cost, int)
                            else _esc(cost or em))
                measured_sev = det.get("severity", "L1")
                llm_sev = r.get("severity_level", em)
                sev_line = "<b>Measured severity:</b> {} ({}) from {} of segmented area".format(
                    SEV_LABEL.get(measured_sev, measured_sev), measured_sev, area_s,
                )
                assessed = "<b>Assessed severity:</b> {} ({})".format(
                    _esc(r.get("severity_label", em)), _esc(llm_sev),
                )
                if llm_sev and llm_sev != measured_sev:
                    assessed += (
                        " &nbsp;<font color='#d35400'>[differs from measurement "
                        "\u2014 area governs, review manually]</font>"
                    )

                origin = ("rule-based template (LLM unavailable)"
                          if r.get("generated_by") == "rule_based_fallback"
                          else "generated assessment")

                lines = [
                    sev_line,
                    assessed,
                    "<b>Urgency:</b> {} days to remediation".format(_esc(r.get("urgency_days", em))),
                    "<b>Description:</b> {}".format(_esc(r.get("description", em))),
                    "<b>Recommended Action:</b> {}".format(_esc(r.get("recommended_action", em))),
                    "<b>Estimated Cost (INR):</b> {}".format(cost_str),
                    "<font size=7 color='#6c757d'>Source: {}</font>".format(origin),
                ]
                for line in lines:
                    report_paras.append(Paragraph(line, S["report"]))
                    report_paras.append(Spacer(1, 2))
            except Exception:
                for line in raw.split("\n"):
                    if line.strip():
                        report_paras.append(Paragraph(line, S["report"]))
        else:
            for line in raw.split("\n"):
                report_paras.append(Paragraph(line, S["report"]) if line.strip()
                                    else Spacer(1, 4))

        report_paras.append(Spacer(1, 14))
        # Keep the header glued to the first few lines; let the rest reflow.
        keep = min(6, len(report_paras))
        flowables.append(KeepTogether(report_paras[:keep]))
        flowables.extend(report_paras[keep:])

    return flowables


# ─────────────────────────────────────────────────────────────────
# Detection coordinate map
# ─────────────────────────────────────────────────────────────────

def _build_map_drawing(detections: list, draw_w: float = 450, draw_h: float = 320) -> Drawing:
    lats = [float(d.get("lat") or 0) for d in detections]
    lons = [float(d.get("lon") or 0) for d in detections]

    pad_frac = 0.15
    if not lats or (max(lats) == min(lats) and max(lons) == min(lons)):
        lat_min = (lats[0] - 0.001) if lats else -0.001
        lat_max = (lats[0] + 0.001) if lats else 0.001
        lon_min = (lons[0] - 0.001) if lons else -0.001
        lon_max = (lons[0] + 0.001) if lons else 0.001
    else:
        lat_span = max(lats) - min(lats) or 0.001
        lon_span = max(lons) - min(lons) or 0.001
        lat_min = min(lats) - lat_span * pad_frac
        lat_max = max(lats) + lat_span * pad_frac
        lon_min = min(lons) - lon_span * pad_frac
        lon_max = max(lons) + lon_span * pad_frac

    margin   = 42          # room for the coordinate labels on both axes
    usable_w = draw_w - 2 * margin
    usable_h = draw_h - 2 * margin

    def to_xy(lat, lon):
        x = margin + (lon - lon_min) / (lon_max - lon_min) * usable_w
        y = margin + (lat - lat_min) / (lat_max - lat_min) * usable_h
        return x, y

    d = Drawing(draw_w, draw_h)
    d.add(Rect(0, 0, draw_w, draw_h,
               fillColor=colors.HexColor("#f0f4f8"), strokeColor=None))

    for i in range(1, 3):
        gx = margin + usable_w * i / 3
        gy = margin + usable_h * i / 3
        d.add(Line(gx, margin, gx, draw_h - margin,
                   strokeColor=colors.HexColor("#d0d8e4"), strokeWidth=0.5))
        d.add(Line(margin, gy, draw_w - margin, gy,
                   strokeColor=colors.HexColor("#d0d8e4"), strokeWidth=0.5))

    d.add(Rect(margin, margin, usable_w, usable_h,
               fillColor=None, strokeColor=colors.HexColor("#aabbcc"), strokeWidth=1))

    for det in detections:
        sev = det.get("severity", "L1")
        x, y = to_xy(float(det.get("lat") or 0), float(det.get("lon") or 0))
        d.add(Circle(x, y, 5, fillColor=SEV_DRAW.get(sev, GREY),
                     strokeColor=WHITE, strokeWidth=1))

    # ── Coordinate labels ────────────────────────────────────────────────
    # The plot previously carried no numbers at all, so a reader could see the
    # spatial arrangement of the defects but not where any of them were.
    axis_col = colors.HexColor("#5a6472")
    for frac in (0.0, 0.5, 1.0):
        lon = lon_min + (lon_max - lon_min) * frac
        lat = lat_min + (lat_max - lat_min) * frac
        x = margin + usable_w * frac
        y = margin + usable_h * frac

        d.add(String(x, margin - 12, f"{lon:.5f}", textAnchor="middle",
                     fontName=FONT, fontSize=6, fillColor=axis_col))
        d.add(String(margin - 4, y - 2, f"{lat:.5f}", textAnchor="end",
                     fontName=FONT, fontSize=6, fillColor=axis_col))

    d.add(String(margin + usable_w / 2, margin - 22, "Longitude (E)",
                 textAnchor="middle", fontName=FONT_BOLD, fontSize=6.5,
                 fillColor=axis_col))

    # Approximate ground scale: 1 degree of latitude is ~111.32 km everywhere.
    span_m = (lat_max - lat_min) * 111_320
    scale_txt = (f"Vertical span \u2248 {span_m:.0f} m"
                 if span_m < 1000 else f"Vertical span \u2248 {span_m / 1000:.2f} km")
    d.add(String(draw_w - margin, draw_h - margin + 6, _t(scale_txt),
                 textAnchor="end", fontName=FONT, fontSize=6.5, fillColor=axis_col))

    # ── Legend, below the plot so it cannot sit on top of a pin ──────────
    legend_x = margin
    for sev, label in [("L3", "HIGH"), ("L2", "MEDIUM"), ("L1", "LOW")]:
        d.add(Circle(legend_x + 4, 12, 4, fillColor=SEV_DRAW[sev], strokeColor=None))
        d.add(String(legend_x + 12, 9, label,
                     fontName=FONT, fontSize=7, fillColor=colors.HexColor("#333")))
        legend_x += 62

    d.add(String(margin + 4, draw_h - margin + 6, "Detection Map",
                 fontName=FONT_BOLD, fontSize=9, fillColor=NAVY))
    return d


# ─────────────────────────────────────────────────────────────────
# Mission summary formatter
# ─────────────────────────────────────────────────────────────────

def _format_mission_summary(raw: str, S: dict):
    raw = str(raw).strip()
    if raw.startswith("{") or raw.startswith("```"):
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw
        try:
            ms = json.loads(raw)
            lines = []
            if ms.get("overall_assessment"):
                lines.append(ms["overall_assessment"])
            if ms.get("most_critical_finding"):
                lines.append("<b>Critical finding:</b> {}".format(ms["most_critical_finding"]))
            if ms.get("recommended_next_inspection"):
                lines.append("<b>Next inspection:</b> {}".format(ms["recommended_next_inspection"]))
            actions = ms.get("priority_actions") or []
            if actions:
                lines.append("<b>Priority actions:</b>")
                for a in actions[:3]:
                    lines.append("\u2022 {}".format(str(a)))
            cell_text = "<br/>".join(lines) if lines else raw[:250]
            return Paragraph(cell_text, S["cell"])
        except Exception:
            pass
    safe = raw[:300].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, S["cell"])


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def generate_inspection_pdf(
    detections: list,
    mission_summary: dict | None = None,
) -> bytes:
    """
    Generate a professional Hawk-I inspection report PDF.

    Args:
        detections:      list of detection dicts from the database
                         (must include raw_box_json, dinov2_flagged, similar_ids,
                          sam_score — all now returned by the updated DB queries).
        mission_summary: optional dict from LLMReporter.batch_report()
                         — adds site health score and LLM summary to cover.

    Returns:
        Raw PDF bytes.
    """
    if not detections:
        detections = []

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=2.8 * cm,
        title="Hawk-I Inspection Report",
        author="Hawk-I Aerial Infrastructure Inspector",
    )

    S   = _styles()
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    total  = len(detections)
    by_sev = Counter(d.get("severity", "L1") for d in detections)

    timestamps = [str(d.get("detected_at", ""))[:19] for d in detections if d.get("detected_at")]
    flight_dur = (f"{timestamps[-1]} → {timestamps[0]}"
                  if len(timestamps) >= 2 else
                  timestamps[0] if timestamps else "—")

    # One list of pairs, not two independent filters: a row carrying a lat but
    # no lon must contribute neither coordinate, or the printed corners pair a
    # value from one detection with a value from another.
    fixes = [(float(d["lat"]), float(d["lon"])) for d in detections
             if d.get("lat") is not None and d.get("lon") is not None]
    lats = [la for la, _ in fixes]
    lons = [lo for _, lo in fixes]
    bbox_str = (
        f"({min(lats):.5f}, {min(lons):.5f}) → ({max(lats):.5f}, {max(lons):.5f})"
        if lats and lons else "—"
    )

    health_score = (mission_summary or {}).get("site_health_score")
    if health_score is None and detections:
        penalty = by_sev.get("L3", 0) * 25 + by_sev.get("L2", 0) * 3 + by_sev.get("L1", 0)
        health_score = max(0, 100 - penalty)

    # Count detections with real SAM-measured area
    sam_measured = sum(
        1 for d in detections
        if (d.get("area_cm2") or 0) > 0 and (d.get("sam_score") or 0) >= 0
    )
    est_area = sum(
        1 for d in detections
        if (d.get("area_cm2") or 0) > 0 and (d.get("sam_score") or 0) < 0
    )

    story: list = []

    # ══ PAGE 1 — Mission Cover ══════════════════════════════════════════════════
    story.append(Paragraph(_esc("HAWK-I STRUCTURAL INSPECTION REPORT"), S["title"]))
    story.append(Paragraph(
        _esc("Aerial Infrastructure Inspector \u00b7 automated defect analysis"),
        S["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=6))
    story.append(Spacer(1, 6))

    meta_rows = [
        ["Field", "Value"],
        ["Report generated",   now],
        ["Flight window",      flight_dur],
        ["GPS bounding box",   bbox_str],
        ["Model stack",        _t("YOLOv11n + YOLO-World \u00b7 SAM 2.1 \u00b7 "
                                    "DINOv2-base \u00b7 Gemma 3")],
        ["Total detections",   str(total)],
        ["Area measurement",   f"{sam_measured} SAM-measured, "
                               f"{est_area} estimated from bounding box"],
        ["Site health score",  f"{health_score}/100" if health_score is not None else "\u2014"],
    ]
    if mission_summary and mission_summary.get("llm_summary"):
        meta_rows.append(["Mission assessment",
                           _format_mission_summary(mission_summary["llm_summary"], S)])

    meta_t = Table(meta_rows, colWidths=[5.5 * cm, 11.5 * cm])
    meta_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("FONTNAME",      (0, 1), (0, -1), FONT_BOLD),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 10))

    # Site health gauge
    if health_score is not None:
        story.append(_build_health_gauge(int(health_score)))
        story.append(Spacer(1, 10))

    # Executive summary table
    story.append(Paragraph("Executive Summary", S["section"]))
    story.append(_summary_table(detections, S))
    story.append(Spacer(1, 10))

    # Confidence band breakdown.  These two blocks used to be separated by an
    # unconditional PageBreak, which stranded the confidence table alone on an
    # otherwise blank page 2.  CondPageBreak only breaks when the next block
    # genuinely will not fit.
    story.append(Paragraph("Confidence Distribution", S["section"]))
    story.append(_confidence_table(detections, S))
    story.append(Spacer(1, 14))

    if detections:
        story.append(CondPageBreak(6 * cm))
        story.append(Paragraph("Detection Statistics", S["section"]))
        story.append(_build_class_chart(detections))
        story.append(Spacer(1, 16))

    story.extend(_defect_detail_cards(detections, S))
    story.append(Spacer(1, 12))

    # Full data table
    story.append(Paragraph("All Detections — Data Table", S["section"]))
    if detections:
        story.append(_detection_table(detections))
    else:
        story.append(Paragraph("No detections recorded.", S["body"]))
    story.append(Spacer(1, 12))

    # Critical LLM analysis
    critical = [d for d in detections if d.get("severity") == "L3" and d.get("llm_report")]
    if critical:
        story.extend(_critical_details(critical, S))

    # ══ Last Page — Detection Map ════════════════════════════════════════════
    if detections:
        story.append(PageBreak())
        story.append(Paragraph("Detection Map", S["section"]))
        _sq = "\u25a0" if UNICODE_OK else "#"
        story.append(Paragraph(
            "Detection pins colour-coded by severity: "
            f"<font color='#e74c3c'>{_sq} HIGH</font>  "
            f"<font color='#e67e22'>{_sq} MEDIUM</font>  "
            f"<font color='#27ae60'>{_sq} LOW</font>",
            S["body"],
        ))
        story.append(Spacer(1, 8))
        story.append(_build_map_drawing(detections))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buffer.getvalue()

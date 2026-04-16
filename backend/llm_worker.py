"""
llm_worker.py — Background LLM reporting job.

Every 30 seconds:
  1. Queries PostGIS for detections in the last 30 s with confidence > 0.60.
  2. Groups by class_name.
  3. For each group, sends a structured prompt to Gemma 3 via Ollama asking for
     {severity_level, severity_label, recommended_action, urgency_days,
      description, estimated_cost_inr} as strict JSON.
  4. Persists the JSON string back into the llm_report column.
  5. Pushes the report entry to all /ws/dashboard WebSocket clients.
"""

import asyncio
import json
import logging
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

from database import get_recent_detections_for_llm, update_detection_report

logger = logging.getLogger(__name__)

OLLAMA_URL  = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}/api/generate"
MODEL_NAME  = "gemma3:4b"
INTERVAL_S  = 30          # seconds between sweeps
CONF_THRESH = 0.60        # minimum confidence to include in LLM sweep

# Set by main.py at startup — reference to the live connected_dashboards set
_dashboard_clients: set = set()


def set_dashboard_clients(clients: set) -> None:
    """Register the live WebSocket client set so reports can be pushed."""
    global _dashboard_clients
    _dashboard_clients = clients


# ── LLM prompting ─────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a structural inspection AI for Indian infrastructure. "
    "Given a group of defect detections, output ONLY a valid JSON object "
    "with exactly these six keys: "
    "severity_level (string: L1/L2/L3), "
    "severity_label (string: Low/Moderate/Critical), "
    "recommended_action (string ≤20 words), "
    "urgency_days (integer: days before action required), "
    "description (string ≤30 words, technical), "
    "estimated_cost_inr (integer: realistic Indian Rupee repair estimate). "
    "No markdown, no code fences — raw JSON only."
)


async def _call_ollama(prompt: str) -> dict:
    """Call Ollama Gemma-3 with a structured-output prompt and parse the JSON."""
    body = {
        "model":   MODEL_NAME,
        "prompt":  f"System: {_SYSTEM}\n\nUser: {prompt}",
        "stream":  False,
        "options": {"temperature": 0.2, "num_predict": 256},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(OLLAMA_URL, json=body)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    return json.loads(raw)


# ── Fallback (rule-based) report when Ollama is unreachable ───────────────────

_COST_BASELINE = {
    "crack":         45_000,
    "spalling":      35_000,
    "corrosion":     55_000,
    "exposed_rebar": 80_000,
    "efflorescence": 18_000,
}


def _fallback(class_name: str, avg_conf: float, count: int) -> dict:
    if avg_conf > 0.85:
        sev_l, sev_s, days = "L3", "Critical", 7
    elif avg_conf > 0.65:
        sev_l, sev_s, days = "L2", "Moderate", 30
    else:
        sev_l, sev_s, days = "L1", "Low", 90

    cost = _COST_BASELINE.get(class_name, 30_000)
    if sev_l == "L3":
        cost = int(cost * 1.5)
    elif sev_l == "L1":
        cost = int(cost * 0.6)

    return {
        "severity_level":     sev_l,
        "severity_label":     sev_s,
        "recommended_action": f"Schedule {class_name} repair within {days} days.",
        "urgency_days":       days,
        "description": (
            f"{count} {class_name} instance(s) detected "
            f"with {avg_conf:.0%} avg confidence. Immediate review advised."
        ),
        "estimated_cost_inr": cost,
    }


# ── Periodic sweep ─────────────────────────────────────────────────────────────

async def _sweep() -> None:
    """Query last 30 s of high-confidence detections, generate per-class reports."""
    rows = await get_recent_detections_for_llm(seconds=INTERVAL_S, min_conf=CONF_THRESH)
    if not rows:
        return

    # Group by class_name
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["class_name"], []).append(row)

    for class_name, dets in groups.items():
        avg_conf = sum(d["confidence"] for d in dets) / len(dets)
        avg_area = sum(d.get("area_cm2", 0.0) for d in dets) / len(dets)
        ids      = [d["id"] for d in dets]
        sevs     = sorted({d["severity"] for d in dets})

        prompt = (
            f"Defect class: {class_name}\n"
            f"Detections in the last 30 s: {len(dets)}\n"
            f"Average confidence: {avg_conf:.2f}\n"
            f"Average area: {avg_area:.1f} cm²\n"
            f"Observed severities: {', '.join(sevs)}\n"
            f"Location: Bengaluru, Karnataka, India\n"
            "Generate the inspection report JSON."
        )

        try:
            report = await _call_ollama(prompt)
        except Exception as exc:
            logger.warning("Ollama call failed for %s: %s — using fallback", class_name, exc)
            report = _fallback(class_name, avg_conf, len(dets))

        report_str = json.dumps(report)

        # Persist to every matched detection row
        for det_id in ids:
            try:
                await update_detection_report(det_id, report_str)
            except Exception as exc:
                logger.error("DB report update failed (id=%d): %s", det_id, exc)

        # Push structured card to dashboard WebSocket clients
        push = {
            "type":       "llm_report",
            "class_name": class_name,
            "det_count":  len(dets),
            "avg_conf":   round(avg_conf, 3),
            **report,
        }
        dead = []
        for client in _dashboard_clients:
            try:
                await client.send_json(push)
            except Exception:
                dead.append(client)
        for c in dead:
            _dashboard_clients.discard(c)

        logger.info(
            "LLM ▶ %s | sev=%s | urgency=%d days | cost=₹%s | %d dets",
            class_name,
            report.get("severity_level", "?"),
            report.get("urgency_days", 0),
            report.get("estimated_cost_inr", "?"),
            len(dets),
        )


async def run_llm_worker() -> None:
    """Infinite background loop — sweeps every INTERVAL_S seconds."""
    logger.info("LLM worker started (interval=%ds, min_conf=%.2f)", INTERVAL_S, CONF_THRESH)
    await asyncio.sleep(INTERVAL_S)   # initial delay so DB has data first
    while True:
        try:
            await _sweep()
        except Exception as exc:
            logger.error("LLM sweep error: %s", exc)
        await asyncio.sleep(INTERVAL_S)

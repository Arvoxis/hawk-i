"""
llm_worker.py — Ollama access layer plus the periodic batch-reporting job.

Two responsibilities:

1. ``call_ollama()`` — the single entry point every module uses to reach the
   LLM.  It wraps the HTTP call in a circuit breaker so that an Ollama that is
   down or misconfigured degrades to a rule-based report *immediately* instead
   of costing LLM_TIMEOUT_S seconds per detection.  That timeout used to be
   paid serially by the processing worker, which throttled the whole pipeline
   to one frame per 30 s whenever Ollama was unhealthy.

2. ``run_llm_worker()`` — a background sweep that runs every
   LLM_BATCH_INTERVAL seconds, groups recent high-confidence detections by
   class, and writes one report per group.  The sweep only fills rows that have
   no report yet; per-detection reports written by processing_worker are richer
   (they carry DINOv2 and similarity context) and are never overwritten.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

import httpx

import config
from database import get_recent_detections_for_llm, update_detection_report

logger = logging.getLogger(__name__)

# Public aliases kept for callers and tests that import them by name.
OLLAMA_URL  = config.OLLAMA_GENERATE_URL
MODEL_NAME  = config.LLM_MODEL
INTERVAL_S  = config.LLM_INTERVAL_S
CONF_THRESH = config.LLM_CONF_THRESHOLD

# Set by main.py at startup — reference to the live connected_dashboards set
_dashboard_clients: set = set()


def set_dashboard_clients(clients: set) -> None:
    """Register the live WebSocket client set so reports can be pushed."""
    global _dashboard_clients
    _dashboard_clients = clients


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a structural inspection AI for Indian infrastructure. "
    "Given a group of defect detections, output ONLY a valid JSON object "
    "with exactly these six keys: "
    "severity_level (string: L1/L2/L3), "
    "severity_label (string: Low/Moderate/Critical), "
    "recommended_action (string <=20 words), "
    "urgency_days (integer: days before action required), "
    "description (string <=30 words, technical), "
    "estimated_cost_inr (integer: realistic Indian Rupee repair estimate). "
    "No markdown, no code fences — raw JSON only."
)

_REQUIRED_KEYS = (
    "severity_level", "severity_label", "recommended_action",
    "urgency_days", "description", "estimated_cost_inr",
)


# ── Circuit breaker ───────────────────────────────────────────────────────────

class _Breaker:
    """Trip after N consecutive failures; stay open for a cooldown period.

    While open, calls return immediately so a dead Ollama costs microseconds
    rather than a full request timeout on every single detection.
    """

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at >= self.cooldown_s:
            # Cooldown elapsed — allow one probe request through.
            self.opened_at = None
            self.failures = 0
            logger.info("LLM circuit breaker: cooldown elapsed, probing Ollama again")
            return False
        return True

    def record_success(self) -> None:
        if self.failures or self.opened_at:
            logger.info("LLM circuit breaker: Ollama healthy again — closing")
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold and self.opened_at is None:
            self.opened_at = time.time()
            logger.error(
                "LLM circuit breaker OPEN after %d consecutive failures — "
                "using rule-based reports for the next %.0fs. "
                "Check that Ollama is reachable at %s and that model %r is pulled. "
                "If the host GPU driver is older than Ollama's CUDA kernels, "
                "set OLLAMA_NUM_GPU=0 in .env to force CPU inference.",
                self.failures, self.cooldown_s, config.OLLAMA_BASE_URL, config.LLM_MODEL,
            )


_breaker = _Breaker(config.LLM_BREAKER_THRESHOLD, config.LLM_BREAKER_COOLDOWN_S)


def breaker_state() -> dict:
    """Expose breaker status for /health and diagnostics."""
    return {
        "open":              _breaker.opened_at is not None,
        "consecutive_fails": _breaker.failures,
        "cooldown_s":        _breaker.cooldown_s,
    }


# ── Raw call ──────────────────────────────────────────────────────────────────

def _parse_response(raw: str,
                    required_keys: tuple[str, ...] = _REQUIRED_KEYS) -> dict:
    """Parse Ollama's text response into the report dict.

    Gemma reliably wraps JSON in ```json fences despite being told not to, so
    the fences are stripped before parsing rather than treated as a failure.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:]
            raw = raw.strip()

    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError(f"expected a JSON object, got {type(report).__name__}")

    missing = [k for k in required_keys if k not in report]
    if missing:
        raise ValueError(f"response missing required keys: {missing}")
    return report


async def _post(
    prompt: str,
    system: str = _SYSTEM,
    required_keys: tuple[str, ...] = _REQUIRED_KEYS,
) -> dict:
    """One HTTP round trip to Ollama. Raises on transport or schema failure."""
    body = {
        "model":   config.LLM_MODEL,
        "prompt":  f"System: {system}\n\nUser: {prompt}",
        "stream":  False,
        "options": config.ollama_options(),
    }
    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_S) as client:
        resp = await client.post(config.OLLAMA_GENERATE_URL, json=body)
        resp.raise_for_status()
        return _parse_response(resp.json().get("response", ""), required_keys)


async def call_ollama(
    prompt: str,
    fallback: Callable[[], dict],
    context: str = "",
    system: str = _SYSTEM,
    required_keys: tuple[str, ...] = _REQUIRED_KEYS,
) -> dict:
    """Generate a report, falling back to a rule-based one on any failure.

    This never raises: callers always get a usable report dict.  ``fallback``
    is a zero-argument callable so the (cheap) fallback is only built when it
    is actually needed.

    ``system``/``required_keys`` default to the per-defect report schema.  The
    mission summary in llm_reporter overrides them so that it runs behind this
    breaker too, instead of paying LLM_TIMEOUT_S against a dead Ollama.
    """
    label = f" [{context}]" if context else ""

    if _breaker.is_open:
        logger.debug("LLM circuit open%s — rule-based report", label)
        return fallback()

    try:
        report = await _post(prompt, system, required_keys)
        _breaker.record_success()
        return report
    except httpx.TimeoutException:
        logger.warning(
            "Ollama timed out after %.0fs%s — rule-based report",
            config.LLM_TIMEOUT_S, label,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200] if exc.response is not None else ""
        logger.warning("Ollama HTTP %s%s: %s — rule-based report",
                       exc.response.status_code, label, detail)
    except (httpx.ConnectError, httpx.TransportError) as exc:
        logger.warning("Ollama unreachable at %s%s: %s — rule-based report",
                       config.OLLAMA_BASE_URL, label, exc)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Ollama returned unusable output%s: %s — rule-based report",
                       label, exc)

    _breaker.record_failure()
    return fallback()


async def health_check() -> bool:
    """Verify Ollama is reachable and the configured model is present."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(config.OLLAMA_TAGS_URL)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception as exc:
        logger.warning("Ollama health check failed (%s at %s) — reports will use "
                       "the rule-based fallback", exc, config.OLLAMA_BASE_URL)
        return False

    if config.LLM_MODEL not in names:
        logger.warning(
            "Ollama is up but model %r is not pulled (available: %s). "
            "Run: ollama pull %s",
            config.LLM_MODEL, ", ".join(sorted(names)) or "none", config.LLM_MODEL,
        )
        return False

    logger.info("Ollama health check passed — model %r available at %s",
                config.LLM_MODEL, config.OLLAMA_BASE_URL)
    return True


# ── Rule-based fallback ───────────────────────────────────────────────────────

# Indicative repair cost per defect class, in INR.  Keys are compared after
# folding away case, spaces and underscores, so "RustStain", "rust_stain" and
# "ruststain" all resolve to the same baseline.
_COST_BASELINE = {
    "crack":                45_000,
    "spalling":             35_000,
    "corrosion":            55_000,
    "exposedreinforcement": 80_000,
    "exposedrebar":         80_000,
    "ruststain":            40_000,
    "scaling":              25_000,
    "delamination":         60_000,
    "efflorescence":        18_000,
}
_DEFAULT_COST = 30_000


def _cost_baseline(class_name: str) -> int:
    key = (class_name or "").lower().replace(" ", "").replace("_", "").replace("-", "")
    return _COST_BASELINE.get(key, _DEFAULT_COST)


def build_fallback(class_name: str, avg_conf: float, count: int) -> dict:
    """Deterministic report used whenever the LLM is unavailable."""
    if avg_conf > 0.85:
        sev_l, sev_s, days = "L3", "Critical", 7
    elif avg_conf > 0.65:
        sev_l, sev_s, days = "L2", "Moderate", 30
    else:
        sev_l, sev_s, days = "L1", "Low", 90

    cost = _cost_baseline(class_name)
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
        "generated_by":       "rule_based_fallback",
    }


# Backwards-compatible private aliases (older call sites imported these names).
_fallback = build_fallback


# ── Periodic sweep ────────────────────────────────────────────────────────────

async def _sweep() -> None:
    """Write a per-class report for recent detections that still lack one."""
    rows = await get_recent_detections_for_llm(
        seconds=config.LLM_INTERVAL_S,
        min_conf=config.LLM_CONF_THRESHOLD,
        only_missing_report=True,
    )
    if not rows:
        return

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["class_name"], []).append(row)

    for class_name, dets in groups.items():
        avg_conf = sum(d["confidence"] for d in dets) / len(dets)
        avg_area = sum(d.get("area_cm2") or 0.0 for d in dets) / len(dets)
        ids      = [d["id"] for d in dets]
        sevs     = sorted({d["severity"] for d in dets if d.get("severity")})

        prompt = (
            f"Defect class: {class_name}\n"
            f"Detections in the last {config.LLM_INTERVAL_S} s: {len(dets)}\n"
            f"Average confidence: {avg_conf:.2f}\n"
            f"Average area: {avg_area:.1f} cm2\n"
            f"Observed severities: {', '.join(sevs) or 'unknown'}\n"
            f"Location: {config.SITE_LOCATION}\n"
            "Generate the inspection report JSON."
        )

        report = await call_ollama(
            prompt,
            fallback=lambda cn=class_name, ac=avg_conf, n=len(dets): build_fallback(cn, ac, n),
            context=f"batch:{class_name}",
        )
        report_str = json.dumps(report)

        for det_id in ids:
            try:
                await update_detection_report(det_id, report_str, only_if_empty=True)
            except Exception as exc:
                logger.error("DB report update failed (id=%d): %s", det_id, exc)

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
            "LLM ▶ %s | sev=%s | urgency=%s days | cost=₹%s | %d dets%s",
            class_name,
            report.get("severity_level", "?"),
            report.get("urgency_days", 0),
            report.get("estimated_cost_inr", "?"),
            len(dets),
            " (fallback)" if report.get("generated_by") == "rule_based_fallback" else "",
        )


async def run_llm_worker() -> None:
    """Infinite background loop — sweeps every LLM_BATCH_INTERVAL seconds."""
    logger.info(
        "LLM worker started (interval=%ds, min_conf=%.2f, model=%s)",
        config.LLM_INTERVAL_S, config.LLM_CONF_THRESHOLD, config.LLM_MODEL,
    )
    await asyncio.sleep(config.LLM_INTERVAL_S)   # initial delay so the DB has data
    while True:
        try:
            await _sweep()
        except Exception as exc:
            logger.error("LLM sweep error: %s", exc)
        await asyncio.sleep(config.LLM_INTERVAL_S)


__all__ = [
    "call_ollama", "build_fallback", "health_check", "breaker_state",
    "run_llm_worker", "set_dashboard_clients",
]

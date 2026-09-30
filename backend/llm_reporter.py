"""
llm_reporter.py — mission-level summary for one inspection session.

Per-detection reports are written by processing_worker and llm_worker.  This
module answers the different, flight-wide question the PDF cover page asks:
how many defects, of what kinds, how healthy is the site, and what should the
engineer do next.

The aggregation (counts, health score, GPS bounding box, worst defect) is pure
arithmetic.  The one-paragraph narrative comes from the LLM -- whichever model
`config.LLM_MODEL` names, `gemma3:4b` by default -- and it goes
through ``llm_worker.call_ollama`` so it shares the circuit breaker with every
other LLM call: when Ollama is down, the PDF endpoint returns immediately with
a rule-based summary instead of blocking for LLM_TIMEOUT_S.

Usage:
    from llm_reporter import LLMReporter
    summary = await LLMReporter().batch_report(rows)
"""

import json
import logging

import config
import llm_worker

logger = logging.getLogger(__name__)

_BATCH_SYSTEM = """You are an expert infrastructure inspector summarizing a drone inspection mission.
Given a list of all detected defects from one flight, output ONLY a valid JSON object with these keys:
  most_critical_finding (string: class name of the worst defect),
  overall_assessment (string ≤30 words),
  recommended_next_inspection (string: date or interval, e.g. "2026-05-01" or "30 days"),
  priority_actions (list of ≤3 strings, each ≤20 words).
No markdown, no code fences — raw JSON only."""

_BATCH_KEYS = (
    "most_critical_finding",
    "overall_assessment",
    "recommended_next_inspection",
    "priority_actions",
)

# Health-score penalty per severity band.  L3 dominates by design: one critical
# defect should drag the score down further than a dozen hairline ones.
_SEVERITY_PENALTY = {"L3": 25, "L2": 3, "L1": 1}
_SEVERITY_RANK = {"L3": 3, "L2": 2, "L1": 1}


class LLMReporter:
    """Aggregates one session's detections into a mission summary."""

    async def batch_report(self, detections: list[dict]) -> dict:
        """
        Generate a mission summary for all detections in one flight session.

        Returns dict with:
            total_defects, by_class, by_severity, site_health_score,
            gps_bbox, most_critical, next_inspection, llm_summary
        """
        if not detections:
            return {
                "total_defects": 0,
                "by_class": {},
                "by_severity": {"L3_high": 0, "L2_medium": 0, "L1_low": 0},
                "site_health_score": 100,
                "gps_bbox": {},
                "most_critical": None,
                "next_inspection": "30 days",
                "llm_summary": "No defects detected this session.",
            }

        by_class: dict[str, int] = {}
        by_sev: dict[str, int] = {}
        for d in detections:
            cls = d.get("class_name", "unknown")
            sev = d.get("severity", "L1")
            by_class[cls] = by_class.get(cls, 0) + 1
            by_sev[sev] = by_sev.get(sev, 0) + 1

        penalty = sum(_SEVERITY_PENALTY.get(s, 1) * n for s, n in by_sev.items())
        health_score = max(0, 100 - penalty)

        # Paired, not two independent filters: a row missing either coordinate
        # contributes neither, so the bbox corners always come from real fixes.
        fixes = [
            (float(d["lat"]), float(d["lon"]))
            for d in detections
            if d.get("lat") is not None and d.get("lon") is not None
        ]
        gps_bbox = {
            "lat_min": min(lat for lat, _ in fixes) if fixes else 0.0,
            "lat_max": max(lat for lat, _ in fixes) if fixes else 0.0,
            "lon_min": min(lon for _, lon in fixes) if fixes else 0.0,
            "lon_max": max(lon for _, lon in fixes) if fixes else 0.0,
        }

        most_critical = max(
            detections,
            key=lambda d: (
                _SEVERITY_RANK.get(d.get("severity", "L1"), 0),
                d.get("confidence", 0),
            ),
        )

        if health_score < 50:
            next_insp = "within 7 days"
        elif health_score < 75:
            next_insp = "within 30 days"
        else:
            next_insp = "within 90 days"

        llm_summary = await self._batch_llm_summary(
            detections,
            by_class,
            by_sev,
            health_score,
            next_insp,
        )

        return {
            "total_defects": len(detections),
            "by_class": by_class,
            "by_severity": {
                "L3_high": by_sev.get("L3", 0),
                "L2_medium": by_sev.get("L2", 0),
                "L1_low": by_sev.get("L1", 0),
            },
            "site_health_score": health_score,
            "gps_bbox": gps_bbox,
            "most_critical": {
                "class_name": most_critical.get("class_name"),
                "severity": most_critical.get("severity"),
                "lat": most_critical.get("lat"),
                "lon": most_critical.get("lon"),
            },
            "next_inspection": next_insp,
            "llm_summary": llm_summary,
        }

    @staticmethod
    async def _batch_llm_summary(
        detections: list[dict],
        by_class: dict,
        by_sev: dict,
        health_score: int,
        next_insp: str,
    ) -> str:
        """Mission narrative as a JSON string — LLM if healthy, rules if not.

        A JSON string rather than a dict because that is what the PDF's
        _format_mission_summary() parses, and what the /api/report response
        serialises.
        """
        prompt = (
            f"Flight session summary:\n"
            f"Total detections: {len(detections)}\n"
            f"Defect classes: {by_class}\n"
            f"Severity counts: {by_sev}\n"
            f"Site health score: {health_score}/100\n"
            f"Location: {config.SITE_LOCATION}\n"
            "Generate mission summary JSON."
        )
        summary = await llm_worker.call_ollama(
            prompt,
            fallback=lambda: _fallback_summary(
                by_class, by_sev, health_score, next_insp
            ),
            context="mission_summary",
            system=_BATCH_SYSTEM,
            required_keys=_BATCH_KEYS,
        )
        return json.dumps(summary)


def _fallback_summary(
    by_class: dict,
    by_sev: dict,
    health_score: int,
    next_insp: str,
) -> dict:
    """Rule-based mission summary, same shape as the LLM's."""
    worst = max(by_sev, key=lambda s: _SEVERITY_RANK.get(s, 0), default="L1")
    worst_class = max(by_class, key=by_class.get, default="unknown")
    total = sum(by_class.values())
    sev_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_sev.items(), reverse=True))

    return {
        "most_critical_finding": worst_class,
        "overall_assessment": (
            f"{total} defect(s) across {len(by_class)} class(es) "
            f"({sev_str}). Site health {health_score}/100."
        ),
        "recommended_next_inspection": next_insp,
        "priority_actions": [
            f"Inspect all {worst} findings on site before repair scheduling.",
            f"Prioritise {worst_class} — the most frequent defect this flight.",
        ],
        "generated_by": "rule_based_fallback",
    }

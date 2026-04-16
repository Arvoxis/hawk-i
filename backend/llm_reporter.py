"""
LLM Report Generator for ZeroDefect / Hawki
--------------------------------------------
Sends detection data to Ollama (Gemma-3 12B) and gets back
a structured infrastructure inspection report.

Usage:
    from llm_reporter import LLMReporter
    reporter = LLMReporter()
    report = await reporter.generate_report(detection_data)
"""

import httpx
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_URL = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}/api/generate"
MODEL_NAME = "gemma3:4b"

SYSTEM_PROMPT = """You are an expert infrastructure inspector analyzing drone-captured defect data.
Given detection details, produce a concise inspection report in EXACTLY this format:

DEFECT: <defect type in plain English>
SEVERITY: <LOW | MEDIUM | HIGH | CRITICAL>
CONFIDENCE: <VERIFIED | FLAGGED_FOR_REVIEW>
AREA: <area_cm2> cm²
LOCATION: <lat>, <lon> at <alt_m>m AGL
PATTERN: <ISOLATED | RECURRING> — <brief explanation if recurring>
URGENCY: <ROUTINE | WITHIN_7_DAYS | WITHIN_48H | IMMEDIATE>
ACTION: <specific repair action with timeframe>
INSPECTOR_NOTE: <one sentence for the human reviewer>

Rules:
- Be specific and actionable — an engineer should act on this immediately
- Severity: area < 100cm² = LOW, 100-500cm² = MEDIUM, 500-2000cm² = HIGH, >2000cm² = CRITICAL
- Adjust severity UP for structural defects (crack, exposed rebar, spalling)
- If DINOv2 flag is set, use CONFIDENCE: FLAGGED_FOR_REVIEW and bump URGENCY up one level
- If similar past defects mentioned, set PATTERN: RECURRING
- Keep the entire report under 180 words
- Do NOT add any preamble or explanation outside the format above"""

_BATCH_SYSTEM = """You are an expert infrastructure inspector summarizing a drone inspection mission.
Given a list of all detected defects from one flight, output ONLY a valid JSON object with these keys:
  most_critical_finding (string: class name of the worst defect),
  overall_assessment (string ≤30 words),
  recommended_next_inspection (string: date or interval, e.g. "2026-05-01" or "30 days"),
  priority_actions (list of ≤3 strings, each ≤20 words).
No markdown, no code fences — raw JSON only."""


class LLMReporter:
    """Generates inspection reports via Ollama + Gemma-3."""

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = MODEL_NAME,
        timeout: float = 30.0,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout

    async def generate_report(self, detection: dict) -> str:
        """
        Generate an inspection report for a single detection.

        Args:
            detection: dict with keys like:
                - detection_class: str ("crack", "spalling", etc.)
                - confidence: float
                - severity: str ("L1", "L2", "L3")
                - area_cm2: float (from SAM 2)
                - lat: float
                - lon: float
                - alt_m: float
                - sam_score: float

        Returns:
            str: formatted inspection report
        """
        prompt = self._build_prompt(detection)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": SYSTEM_PROMPT,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 300,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                report = data.get("response", "").strip()
                logger.info(f"LLM report generated for {detection.get('detection_class', 'unknown')}")
                return report

        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return self._fallback_report(detection)

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama — is it running? (ollama serve)")
            return self._fallback_report(detection)

        except Exception as e:
            logger.error(f"LLM report generation failed: {e}")
            return self._fallback_report(detection)

    def _build_prompt(self, det: dict) -> str:
        """Build the detection summary prompt for the LLM."""
        defect_class  = det.get("detection_class", "unknown defect")
        confidence    = det.get("confidence", 0.0)
        area_cm2      = det.get("area_cm2", 0.0)
        lat           = det.get("lat", 0.0)
        lon           = det.get("lon", 0.0)
        alt_m         = det.get("alt_m", 0.0)
        severity      = det.get("severity", "unknown")
        sam_score     = det.get("sam_score", 0.0)
        dinov2_flagged = det.get("dinov2_flagged", False)
        similar       = det.get("similar_past_detections", [])

        # Source phrasing
        source = det.get("source", "")
        if "grounding_dino" in source or "gdino" in source:
            src_note = "detected by open-vocabulary classifier (verify carefully)"
        else:
            src_note = "detected by primary classifier"

        # DINOv2 flag note
        dino_note = (
            "⚠ DINOv2 similarity check FAILED — verify detection manually"
            if dinov2_flagged
            else "DINOv2 similarity check passed"
        )

        # Similar detections note
        if similar:
            sim_lines = "\n".join(
                f"  • ({s.get('lat',0):.5f}, {s.get('lon',0):.5f}) on "
                f"{str(s.get('detected_at',''))[:10]} "
                f"(similarity={s.get('similarity_score',0):.2f}, area={s.get('area_cm2',0):.1f}cm²)"
                for s in similar[:3]
            )
            sim_note = f"Similar defects found at:\n{sim_lines}\n→ pattern suggests recurring structural issue"
        else:
            sim_note = "No similar past defects found."

        return (
            f"Analyze this infrastructure defect detected by drone:\n"
            f"- Defect type: {defect_class} ({src_note})\n"
            f"- Detection confidence: {confidence:.0%}\n"
            f"- Segmented area: {area_cm2} cm²\n"
            f"- GPS: {lat:.6f}°N, {lon:.6f}°E at {alt_m}m AGL\n"
            f"- Initial severity tag: {severity}\n"
            f"- SAM2 mask confidence: {sam_score:.2f}\n"
            f"- {dino_note}\n"
            f"- {sim_note}\n"
            f"\nGenerate the inspection report."
        )

    @staticmethod
    def _fallback_report(det: dict) -> str:
        """Generate a basic report when LLM is unavailable."""
        defect_class   = det.get("detection_class", "unknown")
        area_cm2       = det.get("area_cm2", 0.0)
        lat            = det.get("lat", 0.0)
        lon            = det.get("lon", 0.0)
        alt_m          = det.get("alt_m", 0.0)
        dinov2_flagged = det.get("dinov2_flagged", False)
        similar        = det.get("similar_past_detections", [])

        if area_cm2 > 2000:
            sev, urgency, action = "CRITICAL", "IMMEDIATE", "Isolate area and emergency repair within 24h"
        elif area_cm2 > 500:
            sev, urgency, action = "HIGH", "WITHIN_48H", "Schedule repair within 1 week"
        elif area_cm2 > 100:
            sev, urgency, action = "MEDIUM", "WITHIN_7_DAYS", "Schedule repair within 1 month"
        else:
            sev, urgency, action = "LOW", "ROUTINE", "Include in next scheduled maintenance"

        confidence_tag = "FLAGGED_FOR_REVIEW" if dinov2_flagged else "VERIFIED"
        pattern = "RECURRING" if similar else "ISOLATED"

        return (
            f"DEFECT: {defect_class}\n"
            f"SEVERITY: {sev}\n"
            f"CONFIDENCE: {confidence_tag}\n"
            f"AREA: {area_cm2} cm²\n"
            f"LOCATION: {lat:.6f}, {lon:.6f} at {alt_m}m AGL\n"
            f"PATTERN: {pattern}\n"
            f"URGENCY: {urgency}\n"
            f"ACTION: {action}\n"
            f"INSPECTOR_NOTE: Report auto-generated (LLM unavailable)"
        )

    async def batch_report(self, detections: list[dict]) -> dict:
        """
        Generate a mission summary for all detections in one flight session.

        Returns dict with:
            total_defects, by_class, by_severity, site_health_score,
            gps_bbox, most_critical, next_inspection, llm_summary
        """
        if not detections:
            return {
                "total_defects":    0,
                "by_class":         {},
                "by_severity":      {},
                "site_health_score": 100,
                "gps_bbox":         {},
                "most_critical":    None,
                "next_inspection":  "30 days",
                "llm_summary":      "No defects detected this session.",
            }

        by_class: dict[str, int] = {}
        by_sev:   dict[str, int] = {}
        for d in detections:
            cls = d.get("class_name", "unknown")
            sev = d.get("severity", "L1")
            by_class[cls] = by_class.get(cls, 0) + 1
            by_sev[sev]   = by_sev.get(sev, 0) + 1

        # Site health score: 100 - penalty (capped at 0)
        penalty = (
            by_sev.get("L3", 0) * 25
            + by_sev.get("L2", 0) * 3
            + by_sev.get("L1", 0) * 1
        )
        health_score = max(0, 100 - penalty)

        lats = [float(d["lat"]) for d in detections if d.get("lat")]
        lons = [float(d["lon"]) for d in detections if d.get("lon")]
        gps_bbox = {
            "lat_min": min(lats) if lats else 0.0,
            "lat_max": max(lats) if lats else 0.0,
            "lon_min": min(lons) if lons else 0.0,
            "lon_max": max(lons) if lons else 0.0,
        }

        # Most critical detection
        sev_order = {"L3": 3, "L2": 2, "L1": 1}
        most_critical = max(
            detections,
            key=lambda d: (sev_order.get(d.get("severity", "L1"), 0), d.get("confidence", 0)),
        )

        # Next inspection recommendation based on health score
        if health_score < 50:
            next_insp = "within 7 days"
        elif health_score < 75:
            next_insp = "within 30 days"
        else:
            next_insp = "within 90 days"

        # LLM summary
        llm_summary = await self._batch_llm_summary(detections, by_class, by_sev, health_score)

        return {
            "total_defects":     len(detections),
            "by_class":          by_class,
            "by_severity":       {
                "L3_high":  by_sev.get("L3", 0),
                "L2_medium": by_sev.get("L2", 0),
                "L1_low":   by_sev.get("L1", 0),
            },
            "site_health_score": health_score,
            "gps_bbox":          gps_bbox,
            "most_critical":     {
                "class_name": most_critical.get("class_name"),
                "severity":   most_critical.get("severity"),
                "lat":        most_critical.get("lat"),
                "lon":        most_critical.get("lon"),
            },
            "next_inspection":   next_insp,
            "llm_summary":       llm_summary,
        }

    async def _batch_llm_summary(
        self,
        detections: list[dict],
        by_class: dict,
        by_sev: dict,
        health_score: int,
    ) -> str:
        """Call the LLM for a mission-level summary, with fallback."""
        prompt = (
            f"Flight session summary:\n"
            f"Total detections: {len(detections)}\n"
            f"Defect classes: {by_class}\n"
            f"Severity counts: {by_sev}\n"
            f"Site health score: {health_score}/100\n"
            f"Location: Bengaluru, Karnataka, India\n"
            "Generate mission summary JSON."
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.ollama_url,
                    json={
                        "model":   self.model,
                        "prompt":  prompt,
                        "system":  _BATCH_SYSTEM,
                        "stream":  False,
                        "options": {"temperature": 0.2, "num_predict": 256},
                    },
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("batch_report LLM call failed: %s — using fallback", exc)
            sev_str = ", ".join(f"{k}: {v}" for k, v in by_sev.items())
            return (
                f"Mission completed. {len(detections)} defects detected "
                f"({sev_str}). Site health: {health_score}/100. "
                f"Review critical findings immediately."
            )

    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                available = any(self.model.split(":")[0] in m for m in models)
                if available:
                    logger.info(f"Ollama OK — {self.model} available")
                else:
                    logger.warning(f"Ollama running but {self.model} not found. Available: {models}")
                return available
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
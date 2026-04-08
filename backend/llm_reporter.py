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
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

SYSTEM_PROMPT = """You are an expert infrastructure inspector analyzing drone-captured defect data.
Given detection details, produce a concise inspection report in EXACTLY this format:

DEFECT: <defect type in plain English>
SEVERITY: <LOW | MEDIUM | HIGH | CRITICAL>
AREA: <area in cm²>
LOCATION: <GPS coordinates>
URGENCY: <one-line urgency assessment>
ACTION: <specific recommended repair action with timeframe>
NOTES: <any additional context based on defect type, size, or location>

Rules:
- Be specific and actionable — an engineer should be able to act on this immediately
- Severity mapping: area < 100cm² = LOW, 100-500cm² = MEDIUM, 500-2000cm² = HIGH, >2000cm² = CRITICAL
- Adjust severity UP if defect type is structural (crack, exposed rebar, spalling)
- Keep the entire report under 150 words
- Do NOT add any preamble or explanation outside the format above"""


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
        defect_class = det.get("detection_class", "unknown defect")
        confidence = det.get("confidence", 0.0)
        area_cm2 = det.get("area_cm2", 0.0)
        lat = det.get("lat", 0.0)
        lon = det.get("lon", 0.0)
        alt_m = det.get("alt_m", 0.0)
        severity = det.get("severity", "unknown")
        sam_score = det.get("sam_score", 0.0)

        return (
            f"Analyze this infrastructure defect detected by drone:\n"
            f"- Defect type: {defect_class}\n"
            f"- Detection confidence: {confidence:.0%}\n"
            f"- Segmented area: {area_cm2} cm²\n"
            f"- GPS: {lat:.6f}°N, {lon:.6f}°E at {alt_m}m altitude\n"
            f"- Initial severity tag: {severity}\n"
            f"- SAM2 mask confidence: {sam_score:.2f}\n"
            f"\nGenerate the inspection report."
        )

    @staticmethod
    def _fallback_report(det: dict) -> str:
        """Generate a basic report when LLM is unavailable."""
        defect_class = det.get("detection_class", "unknown")
        area_cm2 = det.get("area_cm2", 0.0)
        lat = det.get("lat", 0.0)
        lon = det.get("lon", 0.0)

        if area_cm2 > 2000:
            sev = "CRITICAL"
            urgency = "Immediate inspection required"
            action = "Isolate area and schedule emergency repair within 24 hours"
        elif area_cm2 > 500:
            sev = "HIGH"
            urgency = "Priority repair needed"
            action = "Schedule repair within 1 week"
        elif area_cm2 > 100:
            sev = "MEDIUM"
            urgency = "Monitor and plan repair"
            action = "Schedule repair within 1 month"
        else:
            sev = "LOW"
            urgency = "Routine monitoring"
            action = "Include in next scheduled maintenance cycle"

        return (
            f"DEFECT: {defect_class}\n"
            f"SEVERITY: {sev}\n"
            f"AREA: {area_cm2} cm²\n"
            f"LOCATION: {lat:.6f}°N, {lon:.6f}°E\n"
            f"URGENCY: {urgency}\n"
            f"ACTION: {action}\n"
            f"NOTES: Report auto-generated (LLM unavailable)"
        )

    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
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
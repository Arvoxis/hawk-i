"""
test_llm_worker.py — Tests for the LLM resilience layer.

Nothing here talks to Ollama. The point is the behaviour *around* the model:
response parsing, the rule-based fallback, and the circuit breaker that stops
an unhealthy Ollama from serialising the whole pipeline.

That breaker is the reason these tests exist. Before it, the processing worker
awaited the LLM inline per detection with a 30 s timeout and no memory of
failure, so an unreachable Ollama cost 30 s *per detection* — five test frames
produced one stored detection while the system looked like it was working.

    python tests/test_llm_worker.py
"""

import json
import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))

import llm_worker  # noqa: E402


class TestResponseParsing(unittest.TestCase):
    """Gemma wraps JSON in code fences however firmly it is told not to."""

    VALID = {
        "severity_level": "L3",
        "severity_label": "Critical",
        "recommended_action": "Immediate repair.",
        "urgency_days": 7,
        "description": "Significant crack.",
        "estimated_cost_inr": 150000,
    }

    def test_plain_json(self):
        self.assertEqual(llm_worker._parse_response(json.dumps(self.VALID)), self.VALID)

    def test_json_fenced(self):
        raw = "```json\n" + json.dumps(self.VALID) + "\n```"
        self.assertEqual(llm_worker._parse_response(raw), self.VALID)

    def test_bare_fenced(self):
        raw = "```\n" + json.dumps(self.VALID) + "\n```"
        self.assertEqual(llm_worker._parse_response(raw), self.VALID)

    def test_leading_and_trailing_whitespace(self):
        raw = "\n\n  " + json.dumps(self.VALID) + "  \n"
        self.assertEqual(llm_worker._parse_response(raw), self.VALID)

    def test_missing_key_is_rejected(self):
        incomplete = {k: v for k, v in self.VALID.items() if k != "urgency_days"}
        with self.assertRaises(ValueError):
            llm_worker._parse_response(json.dumps(incomplete))

    def test_json_array_is_rejected(self):
        with self.assertRaises(ValueError):
            llm_worker._parse_response("[1, 2, 3]")

    def test_prose_is_rejected(self):
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            llm_worker._parse_response("Sure! Here is the report you asked for.")


class TestFallback(unittest.TestCase):
    """The rule-based report has to be usable, not just non-crashing."""

    def test_has_every_required_key(self):
        report = llm_worker.build_fallback("Crack", 0.9, 3)
        for key in llm_worker._REQUIRED_KEYS:
            self.assertIn(key, report)

    def test_is_attributable(self):
        """A reader must be able to tell template text from model text."""
        report = llm_worker.build_fallback("Crack", 0.9, 1)
        self.assertEqual(report["generated_by"], "rule_based_fallback")

    def test_severity_tracks_confidence(self):
        self.assertEqual(llm_worker.build_fallback("Crack", 0.95, 1)["severity_level"], "L3")
        self.assertEqual(llm_worker.build_fallback("Crack", 0.70, 1)["severity_level"], "L2")
        self.assertEqual(llm_worker.build_fallback("Crack", 0.30, 1)["severity_level"], "L1")

    def test_urgency_tightens_with_severity(self):
        high = llm_worker.build_fallback("Crack", 0.95, 1)["urgency_days"]
        low = llm_worker.build_fallback("Crack", 0.30, 1)["urgency_days"]
        self.assertLess(high, low)

    def test_cost_lookup_ignores_case_and_separators(self):
        """The checkpoint and the query map spell classes differently."""
        base = llm_worker._cost_baseline("RustStain")
        for variant in ("ruststain", "rust_stain", "RUST STAIN", "Rust-Stain"):
            with self.subTest(variant=variant):
                self.assertEqual(llm_worker._cost_baseline(variant), base)

    def test_unknown_class_gets_the_default_cost(self):
        self.assertEqual(llm_worker._cost_baseline("purple polka dots"),
                         llm_worker._DEFAULT_COST)

    def test_parses_as_its_own_output(self):
        """The fallback must satisfy the same schema the LLM path does."""
        report = llm_worker.build_fallback("Spalling", 0.8, 2)
        llm_worker._parse_response(json.dumps(report))


class TestCircuitBreaker(unittest.TestCase):

    def setUp(self):
        self.breaker = llm_worker._Breaker(threshold=3, cooldown_s=60)

    def test_starts_closed(self):
        self.assertFalse(self.breaker.is_open)

    def test_stays_closed_below_threshold(self):
        self.breaker.record_failure()
        self.breaker.record_failure()
        self.assertFalse(self.breaker.is_open)

    def test_opens_at_threshold(self):
        for _ in range(3):
            self.breaker.record_failure()
        self.assertTrue(self.breaker.is_open)

    def test_success_resets_the_count(self):
        self.breaker.record_failure()
        self.breaker.record_failure()
        self.breaker.record_success()
        self.breaker.record_failure()
        self.assertFalse(self.breaker.is_open)

    def test_reopens_only_after_cooldown(self):
        breaker = llm_worker._Breaker(threshold=1, cooldown_s=0.2)
        breaker.record_failure()
        self.assertTrue(breaker.is_open)
        time.sleep(0.25)
        # Cooldown elapsed: one probe request is allowed through.
        self.assertFalse(breaker.is_open)

    def test_probe_failure_reopens_immediately(self):
        breaker = llm_worker._Breaker(threshold=1, cooldown_s=0.2)
        breaker.record_failure()
        time.sleep(0.25)
        self.assertFalse(breaker.is_open)
        breaker.record_failure()
        self.assertTrue(breaker.is_open)

    def test_state_is_reportable(self):
        state = llm_worker.breaker_state()
        for key in ("open", "consecutive_fails", "cooldown_s"):
            self.assertIn(key, state)


class TestCallOllama(unittest.IsolatedAsyncioTestCase):
    """call_ollama must never raise: callers always get a usable report."""

    def setUp(self):
        self._original = llm_worker._breaker
        llm_worker._breaker = llm_worker._Breaker(threshold=2, cooldown_s=60)

    def tearDown(self):
        llm_worker._breaker = self._original

    async def test_open_breaker_short_circuits_without_calling(self):
        called = False

        async def _explode(prompt):
            nonlocal called
            called = True
            raise AssertionError("should not have been called")

        llm_worker._breaker.record_failure()
        llm_worker._breaker.record_failure()
        self.assertTrue(llm_worker._breaker.is_open)

        original_post = llm_worker._post
        llm_worker._post = _explode
        try:
            report = await llm_worker.call_ollama(
                "prompt", fallback=lambda: {"ok": True}, context="test")
        finally:
            llm_worker._post = original_post

        self.assertFalse(called, "an open breaker must not issue a request")
        self.assertEqual(report, {"ok": True})

    async def test_transport_failure_returns_fallback(self):
        async def _fail(prompt):
            raise ValueError("garbage from model")

        original_post = llm_worker._post
        llm_worker._post = _fail
        try:
            report = await llm_worker.call_ollama(
                "prompt",
                fallback=lambda: llm_worker.build_fallback("Crack", 0.9, 1),
                context="test",
            )
        finally:
            llm_worker._post = original_post

        self.assertEqual(report["generated_by"], "rule_based_fallback")

    async def test_success_passes_the_report_through(self):
        expected = dict(TestResponseParsing.VALID)

        async def _ok(prompt):
            return expected

        original_post = llm_worker._post
        llm_worker._post = _ok
        try:
            report = await llm_worker.call_ollama(
                "prompt", fallback=lambda: {"fell": "back"}, context="test")
        finally:
            llm_worker._post = original_post

        self.assertEqual(report, expected)
        self.assertFalse(llm_worker._breaker.is_open)

    async def test_fallback_is_lazy(self):
        """The fallback must not be built on the success path."""
        built = 0

        def _fallback():
            nonlocal built
            built += 1
            return {"fell": "back"}

        async def _ok(prompt):
            return dict(TestResponseParsing.VALID)

        original_post = llm_worker._post
        llm_worker._post = _ok
        try:
            await llm_worker.call_ollama("prompt", fallback=_fallback)
        finally:
            llm_worker._post = original_post

        self.assertEqual(built, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

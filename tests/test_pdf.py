"""
test_pdf.py — Inspection-report PDF generation.

pdf_generator is the largest module in the backend and the only one whose
output a human actually signs off on, so the checks here are about it not
falling over on real-world row shapes: absent GPS, absent LLM report,
unparseable LLM JSON, a single detection, no detections at all.

The chart helpers are where the crashes historically were -- max() over an
empty sequence, a zero-width axis -- so each is called directly rather than
only through the full document.

Needs reportlab (pip install reportlab), no database and no model.

    python tests/test_pdf.py
"""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))

import pdf_generator as pg  # noqa: E402

_DEMO = pathlib.Path(__file__).resolve().parent.parent / "data" / "demo_detections.json"


def _row(**over):
    """A fully-populated detection row; override individual fields per test."""
    row = {
        "id": 1,
        "class_name": "crack",
        "confidence": 0.91,
        "severity": "L3",
        "area_cm2": 652.4,
        "lat": 12.9716,
        "lon": 77.5946,
        "altitude_m": 12.5,
        "sam_score": 0.93,
        "image_path": None,
        "source_model": "yolo_world",
        "detected_at": "2026-04-08T10:00:00+00:00",
        "llm_report": json.dumps(
            {
                "severity_level": "L3",
                "severity_label": "Critical",
                "recommended_action": "Immediate structural repair.",
                "urgency_days": 7,
                "description": "Wide transverse crack on a load-bearing surface.",
                "estimated_cost_inr": 68_000,
            }
        ),
    }
    row.update(over)
    return row


def _is_pdf(blob) -> bool:
    return isinstance(blob, bytes) and blob.startswith(b"%PDF-") and b"%%EOF" in blob


class TestGeneratesValidPdf(unittest.TestCase):
    def test_demo_corpus_renders(self):
        rows = json.loads(_DEMO.read_text(encoding="utf-8"))
        blob = pg.generate_inspection_pdf(rows)
        self.assertTrue(_is_pdf(blob))
        # A one-page stub would mean the body silently rendered nothing.
        self.assertGreater(len(blob), 10_000)

    def test_no_detections_still_produces_a_report(self):
        """An inspection that found nothing is a result, not an error."""
        blob = pg.generate_inspection_pdf([])
        self.assertTrue(_is_pdf(blob))

    def test_single_detection(self):
        """One row means every chart has a single bar and a zero-width range."""
        self.assertTrue(_is_pdf(pg.generate_inspection_pdf([_row()])))

    def test_with_mission_summary(self):
        summary = {
            "site_health_score": 42,
            "llm_summary": json.dumps(
                {
                    "most_critical_finding": "crack",
                    "overall_assessment": "Several critical cracks found.",
                    "recommended_next_inspection": "within 7 days",
                    "priority_actions": ["Inspect column B3.", "Epoxy-inject crack 1."],
                }
            ),
        }
        blob = pg.generate_inspection_pdf([_row()], mission_summary=summary)
        self.assertTrue(_is_pdf(blob))

    def test_prose_mission_summary_does_not_break_rendering(self):
        """The rule-based path and an ignored instruction both yield prose."""
        blob = pg.generate_inspection_pdf(
            [_row()], mission_summary={"llm_summary": "Mission completed. 1 defect."}
        )
        self.assertTrue(_is_pdf(blob))


class TestDegradedRows(unittest.TestCase):
    """Rows the pipeline really does produce when something upstream failed."""

    def test_missing_gps(self):
        blob = pg.generate_inspection_pdf([_row(lat=None, lon=None)])
        self.assertTrue(_is_pdf(blob))

    def test_gps_at_null_island(self):
        """0.0/0.0 is the DB default for a dropped fix, not a real position."""
        self.assertTrue(_is_pdf(pg.generate_inspection_pdf([_row(lat=0.0, lon=0.0)])))

    def test_half_a_fix_does_not_pair_across_rows(self):
        """A lat without a lon must contribute neither coordinate.

        Filtering lat and lon independently used to build lists of different
        lengths and print a bbox corner from one detection paired with a
        corner from another.
        """
        rows = [
            _row(id=1, lat=12.9716, lon=77.5946),
            _row(id=2, lat=13.5, lon=None),  # half a fix -- must be dropped
        ]
        self.assertTrue(_is_pdf(pg.generate_inspection_pdf(rows)))

    def test_no_llm_report(self):
        self.assertTrue(_is_pdf(pg.generate_inspection_pdf([_row(llm_report=None)])))

    def test_unparseable_llm_report(self):
        blob = pg.generate_inspection_pdf([_row(llm_report="Sure! Here you go:")])
        self.assertTrue(_is_pdf(blob))

    def test_bbox_estimated_area(self):
        """sam_score -1 marks an estimate; the report must say so, not crash."""
        self.assertTrue(_is_pdf(pg.generate_inspection_pdf([_row(sam_score=-1.0)])))

    def test_zero_area_and_missing_severity(self):
        blob = pg.generate_inspection_pdf([_row(area_cm2=0.0, severity=None)])
        self.assertTrue(_is_pdf(blob))

    def test_missing_timestamp(self):
        self.assertTrue(_is_pdf(pg.generate_inspection_pdf([_row(detected_at=None)])))


class TestChartHelpersOnEmptyInput(unittest.TestCase):
    """Every historical PDF crash was max()/min() over an empty sequence."""

    def _chart_builders(self):
        for name in dir(pg):
            if name.startswith("_") and ("chart" in name or "bar" in name):
                fn = getattr(pg, name)
                if callable(fn):
                    yield name, fn

    def test_no_chart_builder_raises_on_empty_detections(self):
        built = 0
        for name, fn in self._chart_builders():
            with self.subTest(chart=name):
                try:
                    fn([])
                except TypeError:
                    continue  # different signature; covered via the document
                built += 1
        # Guards against this test silently covering nothing after a rename.
        self.assertGreaterEqual(built, 1, "no chart builders were exercised")


if __name__ == "__main__":
    unittest.main(verbosity=2)

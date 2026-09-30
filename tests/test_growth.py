"""
test_growth.py — Defect-growth comparison across inspections.

The archive lets a new detection be matched against the same physical defect
from an earlier flight, so the report can say whether it is spreading. This
pins the decision logic: what counts as the same defect, what counts as growth,
and which matches are ignored.

No database or model needed.

    python tests/test_growth.py
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))

import processing_worker as pw  # noqa: E402


def _match(
    area,
    sim,
    same_session=False,
    when="2026-05-14 09:03:00",
    session="20260514_090000",
    distance_m=4.0,
    sam_score=0.91,
):
    """An archived match as dinov2_embedder.find_similar returns it.

    Defaults describe the ordinary case: an earlier flight, a few metres away,
    area measured from a SAM mask.
    """
    return {
        "id": 1,
        "session_id": session,
        "same_session": same_session,
        "similarity_score": sim,
        "area_cm2": area,
        "detected_at": when,
        "distance_m": distance_m,
        "sam_score": sam_score,
    }


class TestGrowthClassification(unittest.TestCase):
    def test_growth_above_25pct(self):
        note, growth = pw._describe_growth(341.2, [_match(118.0, 0.87)])
        self.assertEqual(growth["verdict"], "GROWING")
        self.assertAlmostEqual(growth["delta_pct"], 189.2, places=0)
        self.assertIn("GROWING", note)
        self.assertIn("118", note)

    def test_stable_within_25pct(self):
        _, growth = pw._describe_growth(125.0, [_match(118.0, 0.9)])
        self.assertEqual(growth["verdict"], "STABLE")

    def test_large_reduction_is_flagged_not_celebrated(self):
        _, growth = pw._describe_growth(40.0, [_match(118.0, 0.9)])
        self.assertEqual(growth["verdict"], "REDUCED")
        self.assertIn("verify", growth["caveat"])

    def test_verdict_is_an_enum_token_not_prose(self):
        """Callers compare verdict by equality, so it must stay a bare token."""
        for area in (341.2, 125.0, 40.0):
            _, growth = pw._describe_growth(area, [_match(118.0, 0.9)])
            self.assertIn(growth["verdict"], {"GROWING", "STABLE", "REDUCED"})

    def test_boundary_at_exactly_25pct(self):
        _, growth = pw._describe_growth(118.0 * 1.25, [_match(118.0, 0.9)])
        self.assertEqual(growth["verdict"], "GROWING")


class TestSameDefectRequiresColocation(unittest.TestCase):
    """Visual similarity alone must not establish physical identity."""

    def test_far_away_lookalike_is_not_the_same_defect(self):
        far = _match(118.0, 0.95, distance_m=pw._SAME_DEFECT_MAX_DIST_M + 1)
        note, growth = pw._describe_growth(341.0, [far])
        self.assertEqual(note, "")
        self.assertIsNone(growth)

    def test_unknown_distance_is_rejected(self):
        """No fix means no evidence of identity, so no growth claim."""
        note, growth = pw._describe_growth(
            341.0, [_match(118.0, 0.95, distance_m=None)]
        )
        self.assertEqual(note, "")
        self.assertIsNone(growth)

    def test_at_the_distance_boundary_still_counts(self):
        edge = _match(118.0, 0.9, distance_m=pw._SAME_DEFECT_MAX_DIST_M)
        _, growth = pw._describe_growth(341.0, [edge])
        self.assertEqual(growth["verdict"], "GROWING")

    def test_nearby_beats_similar_but_distant(self):
        matches = [
            _match(300.0, 0.99, session="far", distance_m=900.0),
            _match(118.0, 0.84, session="near", distance_m=6.0),
        ]
        _, growth = pw._describe_growth(341.0, matches)
        self.assertEqual(growth["prior_session"], "near")


class TestMeasurementProvenance(unittest.TestCase):
    """A mask area and a bbox estimate are not comparable quantities."""

    def test_bbox_estimated_prior_yields_no_verdict(self):
        note, growth = pw._describe_growth(341.0, [_match(118.0, 0.95, sam_score=-1.0)])
        self.assertIsNone(growth)
        self.assertIn("estimated", note)
        self.assertIn("not comparable", note)

    def test_sam_measured_prior_yields_a_verdict(self):
        _, growth = pw._describe_growth(341.0, [_match(118.0, 0.95, sam_score=0.88)])
        self.assertEqual(growth["verdict"], "GROWING")


class TestMatchSelection(unittest.TestCase):
    def test_ignores_matches_below_similarity_floor(self):
        # 0.55 is similar-looking but not "the same defect"
        note, growth = pw._describe_growth(300.0, [_match(100.0, 0.55)])
        self.assertEqual(note, "")
        self.assertIsNone(growth)

    def test_ignores_same_session_matches(self):
        """A prior *inspection* means a different flight, not this one."""
        note, growth = pw._describe_growth(
            300.0, [_match(100.0, 0.99, same_session=True)]
        )
        self.assertEqual(note, "")
        self.assertIsNone(growth)

    def test_prefers_the_most_similar_prior_match(self):
        matches = [
            _match(100.0, 0.82, session="a"),
            _match(200.0, 0.91, session="b"),  # closest -> should be chosen
            _match(150.0, 0.85, session="c"),
        ]
        _, growth = pw._describe_growth(260.0, matches)
        self.assertEqual(growth["prior_area_cm2"], 200.0)
        self.assertEqual(growth["prior_session"], "b")

    def test_same_session_ignored_even_when_most_similar(self):
        matches = [
            _match(300.0, 0.99, same_session=True),  # closest but same flight
            _match(118.0, 0.83, same_session=False),  # the real prior
        ]
        _, growth = pw._describe_growth(341.0, matches)
        self.assertEqual(growth["prior_area_cm2"], 118.0)

    def test_no_matches_at_all(self):
        note, growth = pw._describe_growth(300.0, [])
        self.assertEqual(note, "")
        self.assertIsNone(growth)

    def test_prior_with_no_area_is_skipped(self):
        note, growth = pw._describe_growth(300.0, [_match(0.0, 0.9)])
        self.assertEqual(note, "")
        self.assertIsNone(growth)

    def test_zero_current_area_yields_nothing(self):
        note, growth = pw._describe_growth(0.0, [_match(118.0, 0.9)])
        self.assertEqual(note, "")
        self.assertIsNone(growth)


class TestHaversine(unittest.TestCase):
    """config.distance_m underpins the co-location gate above."""

    def test_identical_fix_is_zero(self):
        import config

        self.assertAlmostEqual(
            config.distance_m(12.9716, 77.5946, 12.9716, 77.5946), 0.0, places=6
        )

    def test_one_degree_of_latitude_is_about_111km(self):
        import config

        d = config.distance_m(12.0, 77.5946, 13.0, 77.5946)
        self.assertAlmostEqual(d, 111_195, delta=200)

    def test_metre_scale_offset(self):
        """~9e-5 deg of latitude is ~10 m -- the scale the gate operates at."""
        import config

        d = config.distance_m(12.9716, 77.5946, 12.9716 + 9e-5, 77.5946)
        self.assertAlmostEqual(d, 10.0, delta=0.2)


class TestAreaEstimateSharesConfigFormula(unittest.TestCase):
    """The bbox fallback must use the same GSD as the SAM path."""

    def test_bbox_estimate_matches_config(self):
        import config

        box = [0, 0, 100, 50]  # 5000 px²
        est = pw._estimate_area_cm2_from_box(box, 10.0, 1920)
        self.assertAlmostEqual(est, config.px_to_cm2(5000, 10.0, 1920), places=2)

    def test_bad_altitude_uses_default(self):
        import config

        box = [0, 0, 100, 50]
        est = pw._estimate_area_cm2_from_box(box, 0.0, 1920)
        expected = config.px_to_cm2(5000, config.DEFAULT_ALT_M, 1920)
        self.assertAlmostEqual(est, expected, places=2)

    def test_degenerate_box_is_zero(self):
        self.assertEqual(pw._estimate_area_cm2_from_box([10, 10, 10, 10], 10.0), 0.0)
        self.assertEqual(pw._estimate_area_cm2_from_box([], 10.0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

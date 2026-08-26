"""
test_geometry.py — Unit tests for the GSD / area / severity maths.

These run without a GPU, a model file, a database, or a running backend.

Why they exist: the ground-sampling-distance conversion was previously written
out three times (sam3_worker, processing_worker, sam2_segmenter) with two
different unit factors, and the factor in use was 10x too small — which made
every reported defect area 100x too small and pinned almost everything to
severity L1.  These tests pin the conversion to an independently derived
value so the regression cannot come back silently.

    python tests/test_geometry.py
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))

import config  # noqa: E402


class TestGSD(unittest.TestCase):
    """Ground Sampling Distance: how many centimetres one pixel covers."""

    def test_matches_hand_computed_value(self):
        """IMX477 at 10 m over a 1920 px frame is ~0.69 cm/px.

        Derivation, independent of the implementation:
            ground_width_m = alt_m * sensor_width_mm / focal_mm
                           = 10 * 6.287 / 4.74            = 13.2637 m
            cm per pixel   = 13.2637 m * 100 / 1920 px    = 0.6908 cm/px
        """
        gsd = config.gsd_cm_per_px(10.0, 1920)
        self.assertAlmostEqual(gsd, 0.6908, places=3)

    def test_is_metres_times_one_hundred(self):
        """The cm figure must be exactly 100x the raw metres-per-pixel ratio."""
        alt_m, width_px = 12.5, 1280
        metres_per_px = (
            alt_m * config.CAMERA_SENSOR_WIDTH_MM
        ) / (config.CAMERA_FOCAL_MM * width_px)
        self.assertAlmostEqual(
            config.gsd_cm_per_px(alt_m, width_px), metres_per_px * 100.0, places=9
        )

    def test_scales_linearly_with_altitude(self):
        """Twice the altitude covers twice the ground per pixel."""
        self.assertAlmostEqual(
            config.gsd_cm_per_px(20.0, 1920),
            2 * config.gsd_cm_per_px(10.0, 1920),
            places=9,
        )

    def test_scales_inversely_with_frame_width(self):
        """A frame with twice the pixels resolves twice as finely."""
        self.assertAlmostEqual(
            config.gsd_cm_per_px(10.0, 3840),
            config.gsd_cm_per_px(10.0, 1920) / 2,
            places=9,
        )

    def test_rejects_non_positive_width(self):
        with self.assertRaises(ValueError):
            config.gsd_cm_per_px(10.0, 0)


class TestAreaConversion(unittest.TestCase):

    def test_area_scales_with_gsd_squared(self):
        """Area is a two-dimensional quantity: doubling GSD quadruples cm2."""
        near = config.px_to_cm2(10_000, 10.0, 1920)
        far  = config.px_to_cm2(10_000, 20.0, 1920)
        self.assertAlmostEqual(far / near, 4.0, places=6)

    def test_known_region(self):
        """A 100x100 px region at 10 m covers ~47.7 cm2.

            0.6908 cm/px squared = 0.4772 cm2/px, times 10 000 px = 4771.6 cm2 / 100
        """
        self.assertAlmostEqual(config.px_to_cm2(10_000, 10.0, 1920), 4771.6, delta=1.0)

    def test_zero_pixels_is_zero_area(self):
        self.assertEqual(config.px_to_cm2(0, 10.0, 1920), 0.0)

    def test_a_full_frame_defect_is_a_plausible_size(self):
        """Sanity floor: a defect filling a 1920x1080 frame at 10 m must be
        on the order of square metres, not square millimetres.

        This is the assertion that would have caught the original 100x error —
        under the old formula a full frame measured ~99 cm2, roughly a
        postcard, for a region genuinely about 13 m across.
        """
        area_cm2 = config.px_to_cm2(1920 * 1080, 10.0, 1920)
        self.assertGreater(area_cm2, 500_000)    # > 50 m2
        self.assertLess(area_cm2, 5_000_000)     # < 500 m2


class TestAltitudeResolution(unittest.TestCase):

    def test_good_altitude_passes_through(self):
        alt, substituted = config.resolve_altitude(12.5)
        self.assertEqual(alt, 12.5)
        self.assertFalse(substituted)

    def test_none_is_substituted(self):
        alt, substituted = config.resolve_altitude(None)
        self.assertEqual(alt, config.DEFAULT_ALT_M)
        self.assertTrue(substituted)

    def test_below_minimum_is_substituted(self):
        """A sub-2 m reading means no GPS lock, not a drone hovering at 1 m."""
        alt, substituted = config.resolve_altitude(0.0)
        self.assertEqual(alt, config.DEFAULT_ALT_M)
        self.assertTrue(substituted)

    def test_garbage_is_substituted(self):
        alt, substituted = config.resolve_altitude("not-a-number")
        self.assertEqual(alt, config.DEFAULT_ALT_M)
        self.assertTrue(substituted)


class TestSeverityClassification(unittest.TestCase):

    def test_boundaries(self):
        self.assertEqual(config.classify_severity(0.0), "L1")
        self.assertEqual(config.classify_severity(config.SEVERITY_L2_CM2 - 0.01), "L1")
        self.assertEqual(config.classify_severity(config.SEVERITY_L2_CM2), "L2")
        self.assertEqual(config.classify_severity(config.SEVERITY_L3_CM2 - 0.01), "L2")
        self.assertEqual(config.classify_severity(config.SEVERITY_L3_CM2), "L3")
        self.assertEqual(config.classify_severity(10_000.0), "L3")

    def test_thresholds_are_ordered(self):
        self.assertLess(config.SEVERITY_L2_CM2, config.SEVERITY_L3_CM2)


class TestWeightsResolution(unittest.TestCase):

    def test_falls_back_to_a_named_path(self):
        """A missing checkpoint still yields a path, so logs can name it."""
        path = config.resolve_weights("definitely-not-here.pt")
        self.assertEqual(path.name, "definitely-not-here.pt")
        self.assertFalse(path.exists())

    def test_env_override_wins(self, ):
        import os
        os.environ["HAWKI_TEST_WEIGHTS"] = "/tmp/custom.pt"
        try:
            path = config.resolve_weights("hawki_yolo11n.pt", "HAWKI_TEST_WEIGHTS")
            self.assertEqual(str(path).replace("\\", "/"), "/tmp/custom.pt")
        finally:
            del os.environ["HAWKI_TEST_WEIGHTS"]


class TestOllamaOptions(unittest.TestCase):

    def test_num_gpu_omitted_when_unset(self):
        """-1 means 'let Ollama decide' and must not appear in the payload."""
        original = config.OLLAMA_NUM_GPU
        config.OLLAMA_NUM_GPU = -1
        try:
            self.assertNotIn("num_gpu", config.ollama_options())
        finally:
            config.OLLAMA_NUM_GPU = original

    def test_num_gpu_forwarded_when_set(self):
        original = config.OLLAMA_NUM_GPU
        config.OLLAMA_NUM_GPU = 0
        try:
            self.assertEqual(config.ollama_options()["num_gpu"], 0)
        finally:
            config.OLLAMA_NUM_GPU = original

    def test_overrides_apply(self):
        self.assertEqual(config.ollama_options(num_predict=42)["num_predict"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)

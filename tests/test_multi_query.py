"""
test_multi_query.py — Unit tests for multi_query_yoloworld.py

All tests run without a real YOLO-World model or GPU.
Pure-Python helpers (_build_reverse_map, _iou, _nms_pure, _nms_per_class) are
tested directly.  MultiQueryYOLOWorld.predict() is tested via a mock that
replicates the ultralytics result structure.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub ultralytics so the module imports cleanly without it installed
# ---------------------------------------------------------------------------
_ultralytics_stub = types.ModuleType("ultralytics")
_ultralytics_stub.YOLOWorld = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("ultralytics", _ultralytics_stub)

from multi_query_yoloworld import (   # noqa: E402 — must come after stub
    QUERY_MAP,
    Detection,
    MultiQueryYOLOWorld,
    _build_reverse_map,
    _iou,
    _nms_per_class,
    _nms_pure,
)


# ---------------------------------------------------------------------------
# 1. Reverse-map correctness
# ---------------------------------------------------------------------------

class TestReverseMap(unittest.TestCase):

    def setUp(self):
        self.reverse = _build_reverse_map(QUERY_MAP)

    def test_known_queries_map_to_correct_class(self):
        cases = {
            "thin line in concrete":          "Crack",
            "hairline crack":                 "Crack",
            "horizontal crack in concrete":   "Crack",
            "brown stain on concrete":        "RustStain",
            "reddish discoloration":          "RustStain",
            "bare metal rod":                 "Exposed_reinforcement",
            "corroded steel bar":             "Exposed_reinforcement",
            "peeling concrete surface":       "Scaling",
            "rough eroded surface":           "Scaling",
            "broken concrete chunk":          "Spalling",
            "hollow area in wall":            "Spalling",
            "white powder on wall":           "Efflorescence",
            "mineral deposit on brick":       "Efflorescence",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.reverse[query], expected)

    def test_all_queries_present(self):
        total_queries = sum(len(qs) for qs in QUERY_MAP.values())
        self.assertEqual(len(self.reverse), total_queries)

    def test_duplicate_query_warns(self):
        dup_map = {
            "ClassA": ["shared query", "unique a"],
            "ClassB": ["shared query", "unique b"],
        }
        with self.assertLogs("multi_query_yoloworld", level="WARNING") as cm:
            reverse = _build_reverse_map(dup_map)
        # First mapping wins
        self.assertEqual(reverse["shared query"], "ClassA")
        self.assertTrue(any("shared query" in line for line in cm.output))


# ---------------------------------------------------------------------------
# 2. IoU helper
# ---------------------------------------------------------------------------

class TestIoU(unittest.TestCase):

    def test_identical_boxes_iou_is_one(self):
        box = [0.0, 0.0, 100.0, 100.0]
        self.assertAlmostEqual(_iou(box, box), 1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        self.assertAlmostEqual(_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_partial_overlap(self):
        # Two 100×100 boxes offset by 50 → 50×50 intersection = 2500
        # union = 100*100 + 100*100 - 2500 = 17500
        iou = _iou([0, 0, 100, 100], [50, 50, 150, 150])
        self.assertAlmostEqual(iou, 2500 / 17500, places=5)


# ---------------------------------------------------------------------------
# 3. NMS — same-class merging
# ---------------------------------------------------------------------------

class TestNMS(unittest.TestCase):

    def _crack(self, conf, x_offset=0):
        return Detection(
            label="Crack",
            confidence=conf,
            bbox=[100 + x_offset, 100, 200 + x_offset, 200],
            query_matched="hairline crack",
        )

    def test_high_overlap_same_class_merged(self):
        """Two nearly identical Crack boxes → only higher-confidence survives."""
        det_high = Detection("Crack", 0.90, [100, 100, 200, 200], "hairline crack")
        det_low  = Detection("Crack", 0.70, [102, 102, 202, 202], "thin line in concrete")
        result = _nms_pure([det_high, det_low], iou_threshold=0.4)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.90)

    def test_low_overlap_same_class_both_kept(self):
        """Two Crack boxes far apart → both survive."""
        det_a = self._crack(0.85, x_offset=0)
        det_b = self._crack(0.80, x_offset=300)   # no overlap
        result = _nms_pure([det_a, det_b], iou_threshold=0.4)
        self.assertEqual(len(result), 2)

    def test_nms_keeps_highest_confidence(self):
        dets = [
            Detection("Crack", 0.60, [100, 100, 200, 200], "q1"),
            Detection("Crack", 0.95, [101, 101, 201, 201], "q2"),
            Detection("Crack", 0.75, [100, 100, 200, 200], "q3"),
        ]
        result = _nms_pure(dets, iou_threshold=0.4)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.95)


# ---------------------------------------------------------------------------
# 4. Per-class NMS — different classes must NOT be merged
# ---------------------------------------------------------------------------

class TestNMSPerClass(unittest.TestCase):

    def test_identical_boxes_different_classes_both_kept(self):
        """A Crack and a Spalling box at the same location must both survive."""
        det_crack  = Detection("Crack",    0.90, [100, 100, 200, 200], "thin line in concrete")
        det_spall  = Detection("Spalling", 0.85, [100, 100, 200, 200], "broken concrete chunk")
        result = _nms_per_class([det_crack, det_spall], iou_threshold=0.4)
        labels = {d.label for d in result}
        self.assertEqual(labels, {"Crack", "Spalling"})
        self.assertEqual(len(result), 2)

    def test_three_classes_independent_nms(self):
        """High-overlap pairs within each of three classes → one survivor per class."""
        dets = [
            # Crack pair — high overlap
            Detection("Crack",        0.90, [10, 10, 50, 50], "hairline crack"),
            Detection("Crack",        0.70, [11, 11, 51, 51], "fracture in wall"),
            # RustStain pair — high overlap
            Detection("RustStain",    0.88, [10, 10, 50, 50], "brown stain on concrete"),
            Detection("RustStain",    0.65, [10, 10, 50, 50], "rust mark on surface"),
            # Spalling — single box, survives unconditionally
            Detection("Spalling",     0.80, [200, 200, 300, 300], "broken concrete chunk"),
        ]
        result = _nms_per_class(dets, iou_threshold=0.4)
        # One per class
        by_label: dict[str, list] = {}
        for d in result:
            by_label.setdefault(d.label, []).append(d)

        self.assertEqual(len(by_label["Crack"]),     1)
        self.assertEqual(len(by_label["RustStain"]), 1)
        self.assertEqual(len(by_label["Spalling"]),  1)
        # Best confidence kept within each class
        self.assertAlmostEqual(by_label["Crack"][0].confidence,     0.90)
        self.assertAlmostEqual(by_label["RustStain"][0].confidence, 0.88)


# ---------------------------------------------------------------------------
# 5. MultiQueryYOLOWorld.predict() integration (mocked model)
# ---------------------------------------------------------------------------

def _make_mock_box(cls_idx: int, conf: float, xyxy: list[float]):
    """Build a minimal mock that looks like an ultralytics Boxes entry."""
    import torch
    box = MagicMock()
    box.cls  = torch.tensor([cls_idx], dtype=torch.float32)
    box.conf = torch.tensor([conf],    dtype=torch.float32)
    box.xyxy = torch.tensor([xyxy],    dtype=torch.float32)
    return box


class TestMultiQueryYOLOWorldPredict(unittest.TestCase):

    def _make_detector(self, mock_boxes: list, names: dict[int, str]) -> MultiQueryYOLOWorld:
        """Instantiate detector with a mocked YOLOWorld model."""
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = names

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with patch("multi_query_yoloworld._YOLOWorld", return_value=mock_model):
            detector = MultiQueryYOLOWorld("fake.pt", conf=0.25, iou_nms=0.4)
        return detector

    def test_single_detection_remapped_correctly(self):
        names = {0: "hairline crack"}
        boxes = [_make_mock_box(0, 0.88, [10, 10, 50, 50])]
        detector = self._make_detector(boxes, names)

        result = detector.predict(MagicMock())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "Crack")
        self.assertEqual(result[0].query_matched, "hairline crack")
        self.assertAlmostEqual(result[0].confidence, 0.88)

    def test_two_same_class_high_overlap_merged(self):
        names = {0: "hairline crack", 1: "thin line in concrete"}
        boxes = [
            _make_mock_box(0, 0.90, [10.0, 10.0, 50.0, 50.0]),
            _make_mock_box(1, 0.70, [11.0, 11.0, 51.0, 51.0]),
        ]
        detector = self._make_detector(boxes, names)
        result = detector.predict(MagicMock())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "Crack")
        self.assertAlmostEqual(result[0].confidence, 0.90)

    def test_two_different_classes_both_survive(self):
        names = {0: "hairline crack", 1: "broken concrete chunk"}
        boxes = [
            _make_mock_box(0, 0.90, [10.0, 10.0, 50.0, 50.0]),
            _make_mock_box(1, 0.85, [10.0, 10.0, 50.0, 50.0]),
        ]
        detector = self._make_detector(boxes, names)
        result = detector.predict(MagicMock())
        labels = {d.label for d in result}
        self.assertIn("Crack",    labels)
        self.assertIn("Spalling", labels)
        self.assertEqual(len(result), 2)

    def test_results_sorted_by_confidence_descending(self):
        names = {0: "hairline crack", 1: "brown stain on concrete"}
        boxes = [
            _make_mock_box(0, 0.60, [10.0, 10.0, 50.0, 50.0]),
            _make_mock_box(1, 0.92, [200.0, 200.0, 300.0, 300.0]),
        ]
        detector = self._make_detector(boxes, names)
        result = detector.predict(MagicMock())
        self.assertGreaterEqual(result[0].confidence, result[1].confidence)

    def test_unknown_query_in_result_is_skipped(self):
        names = {0: "completely unknown phrase"}
        boxes = [_make_mock_box(0, 0.80, [10.0, 10.0, 50.0, 50.0])]
        detector = self._make_detector(boxes, names)
        with self.assertLogs("multi_query_yoloworld", level="WARNING"):
            result = detector.predict(MagicMock())
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()

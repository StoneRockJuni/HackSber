"""Unit tests for the SberStroi YOLO adapter (no model weights required)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sber_vision import (
    CLASS_NAMES,
    INPUT_SIZE,
    _decode_output,
    _letterbox,
    analyze_construction,
    detect_elements,
    model_available,
    stage_from_classes,
)

import numpy as np


class SberVisionTests(unittest.TestCase):
    def test_class_order_matches_dataset(self) -> None:
        self.assertEqual(
            CLASS_NAMES,
            ["doors", "fence", "foundation", "roof_finished", "walls", "windows"],
        )

    def test_stage_from_classes(self) -> None:
        cases = [
            ({"foundation"}, "50%"),
            ({"walls"}, "50%"),
            ({"foundation", "walls"}, "50%"),
            ({"roof_finished"}, "50%"),  # ambiguous: neighbour roof at frame edge
            ({"roof_finished", "foundation"}, "50%"),
            ({"roof_finished", "walls"}, "100%"),
            ({"roof_finished", "windows"}, "100%"),
            ({"roof_finished", "doors"}, "100%"),
            ({"roof_finished", "walls", "windows", "doors"}, "100%"),
            ({"windows", "doors"}, "100%"),
            ({"windows"}, "0%"),
            ({"fence"}, "0%"),
            (set(), "0%"),
        ]
        for classes, expected in cases:
            with self.subTest(classes=classes):
                stage, confidence = stage_from_classes(classes, {name: 0.8 for name in classes})
                self.assertEqual(stage, expected)
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)

    def test_only_closed_house_is_auto_confident(self) -> None:
        _, walls_conf = stage_from_classes({"walls"}, {"walls": 0.95})
        _, closed_conf = stage_from_classes(
            {"roof_finished", "walls"}, {"roof_finished": 0.95, "walls": 0.9}
        )
        self.assertLess(walls_conf, 0.75)
        self.assertGreaterEqual(closed_conf, 0.75)

    def test_no_detections_means_manual_review(self) -> None:
        stage, confidence = stage_from_classes(set())
        self.assertEqual(stage, "0%")
        self.assertEqual(confidence, 0.0)

    def test_fence_only_is_not_confident_zero(self) -> None:
        _, confidence = stage_from_classes({"fence"}, {"fence": 0.9})
        self.assertLess(confidence, 0.55)

    def test_letterbox_is_square(self) -> None:
        image = np.zeros((480, 800, 3), dtype=np.uint8)
        canvas, scale, dx, dy = _letterbox(image, INPUT_SIZE)
        self.assertEqual(canvas.shape, (INPUT_SIZE, INPUT_SIZE, 3))
        self.assertGreater(scale, 0)
        self.assertGreaterEqual(dx, 0)
        self.assertGreaterEqual(dy, 0)

    def test_decode_output_yolov8_layout(self) -> None:
        # YOLOv8 ONNX layout: (1, 4 + classes, anchors). Anchor 0 is a strong
        # "windows" detection; anchor 1 is below the confidence threshold.
        output = np.zeros((1, 4 + len(CLASS_NAMES), 2), dtype=np.float32)
        output[0, :4, 0] = [320, 320, 100, 100]
        output[0, 4 + 5, 0] = 0.9
        output[0, :4, 1] = [100, 100, 10, 10]
        output[0, 4 + 2, 1] = 0.1
        result = _decode_output(
            output,
            scale=1.0,
            dx=0,
            dy=0,
            original_width=640,
            original_height=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class_name"], "windows")
        self.assertGreater(result[0]["confidence"], 0.8)

    def test_missing_model_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.onnx")
            with patch.dict(os.environ, {"SBER_YOLO_MODEL": missing}):
                self.assertFalse(model_available())
                self.assertIsNone(detect_elements(b"not-an-image"))
                self.assertIsNone(analyze_construction(b"not-an-image"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

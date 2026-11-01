"""Тесты объект-центричной логики стадии и слияния серии.

Модель здесь не нужна: детектор подменяется управляемыми боксами, поэтому
проверяется именно бизнес-логика - центральность объекта, приоритет кровли,
консервативность 0% и поведение fusion по нескольким кадрам.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import photo_ai


def fake_detector(boxes):
    """boxes: {class_name: (cx, cy, width, height, score)} в долях кадра."""

    def detector(image, conf_threshold=0.25, iou_threshold=0.45):
        height, width = image.shape[:2]
        result = []
        for class_name, (cx, cy, bw, bh, score) in boxes.items():
            result.append(
                {
                    "class_name": class_name,
                    "class_name_ru": class_name,
                    "confidence": score,
                    "box": [
                        int((cx - bw / 2) * width),
                        int((cy - bh / 2) * height),
                        int((cx + bw / 2) * width),
                        int((cy + bh / 2) * height),
                    ],
                }
            )
        return result

    return detector


def blank_image(width: int = 640, height: int = 480, colour=(60, 60, 60)) -> np.ndarray:
    return np.full((height, width, 3), colour, dtype=np.uint8)


class PhotoAiFrameTests(unittest.TestCase):
    def _analyze(self, boxes, image=None):
        image = image if image is not None else blank_image()
        with mock.patch.object(photo_ai, "model_available", return_value=True), mock.patch.object(
            photo_ai, "detect_elements_array", side_effect=fake_detector(boxes)
        ):
            return photo_ai.analyze_frame(image)

    def test_central_roof_is_confident_100(self) -> None:
        frame = self._analyze({"roof_finished": (0.5, 0.45, 0.45, 0.30, 0.9)})
        self.assertEqual(frame["stage"], "100%")
        self.assertGreaterEqual(frame["confidence"], photo_ai.MANUAL_REVIEW_THRESHOLD)
        self.assertFalse(frame["needs_manual_review"])
        self.assertEqual(frame["reason"], "central_finished_roof")

    def test_neighbour_roof_at_the_edge_is_not_100(self) -> None:
        frame = self._analyze({"roof_finished": (0.92, 0.5, 0.12, 0.10, 0.95)})
        self.assertNotEqual(frame["stage"], "100%")

    def test_central_foundation_is_50_and_review(self) -> None:
        frame = self._analyze({"foundation": (0.5, 0.5, 0.4, 0.3, 0.8)})
        self.assertEqual(frame["stage"], "50%")
        self.assertTrue(frame["needs_manual_review"])
        self.assertLess(frame["confidence"], photo_ai.MANUAL_REVIEW_THRESHOLD)

    def test_walls_without_roof_is_50(self) -> None:
        frame = self._analyze({"walls": (0.5, 0.5, 0.35, 0.25, 0.9)})
        self.assertEqual(frame["stage"], "50%")

    def test_openings_only_never_claim_100(self) -> None:
        frame = self._analyze({"windows": (0.5, 0.5, 0.2, 0.1, 0.95)})
        self.assertEqual(frame["stage"], "50%")
        self.assertTrue(frame["needs_manual_review"])

    def test_green_empty_plot_is_low_confidence_zero(self) -> None:
        frame = self._analyze({}, image=blank_image(colour=(0, 180, 0)))
        self.assertEqual(frame["stage"], "0%")
        self.assertLessEqual(frame["confidence"], 0.6)

    def test_undecodable_frame_does_not_crash(self) -> None:
        frame = photo_ai.analyze_frame(b"not an image at all")
        self.assertEqual(frame["stage"], "0%")
        self.assertEqual(frame["confidence"], 0.0)
        self.assertEqual(frame["reason"], "frame_not_decoded")

    def test_render_overlay_keeps_geometry(self) -> None:
        frame = self._analyze({"roof_finished": (0.5, 0.45, 0.45, 0.30, 0.9)})
        overlay = photo_ai.render_overlay(blank_image(800, 600), frame)
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape[:2], (600, 800))


class PhotoAiFusionTests(unittest.TestCase):
    @staticmethod
    def frame(stage: str, confidence: float) -> dict:
        return {"stage": stage, "confidence": confidence, "classes": [], "detections": []}

    def test_strong_100_wins_over_empty_frames(self) -> None:
        fused = photo_ai.fuse_frames(
            [self.frame("100%", 0.90), self.frame("0%", 0.55), self.frame("0%", 0.40)]
        )
        self.assertEqual(fused["stage"], "100%")
        self.assertFalse(fused["needs_manual_review"])

    def test_two_weak_roof_frames_give_100_with_review(self) -> None:
        fused = photo_ai.fuse_frames([self.frame("100%", 0.55), self.frame("100%", 0.60)])
        self.assertEqual(fused["stage"], "100%")
        self.assertTrue(fused["needs_manual_review"])

    def test_unfinished_consensus_is_50(self) -> None:
        fused = photo_ai.fuse_frames(
            [self.frame("50%", 0.45), self.frame("50%", 0.45), self.frame("0%", 0.40)]
        )
        self.assertEqual(fused["stage"], "50%")
        self.assertTrue(fused["needs_manual_review"])

    def test_single_weak_roof_frame_is_not_zero(self) -> None:
        """Один кадр готового дома не должен превращаться в «ничего не найдено».

        Регрессия: раньше одиночный кадр со стадией 100 % и уверенностью 0.59
        проваливался в ветку «0 %», и пользователь, загрузивший одно фото
        готового дома, получал «повторный вылет».
        """
        fused = photo_ai.fuse_frames([self.frame("100%", 0.59)])
        self.assertEqual(fused["stage"], "100%")
        self.assertAlmostEqual(fused["confidence"], 0.59, places=2)
        self.assertTrue(fused["needs_manual_review"])

    def test_roof_outvoted_by_unfinished_frames(self) -> None:
        """Один кадр с «кровлей» не должен перебивать три кадра с фундаментом.

        Именно так ловилась соседская кровля на стройплощадке.
        """
        fused = photo_ai.fuse_frames(
            [
                self.frame("50%", 0.45),
                self.frame("50%", 0.45),
                self.frame("50%", 0.45),
                self.frame("100%", 0.90),
            ]
        )
        self.assertEqual(fused["stage"], "50%")
        self.assertTrue(fused["needs_manual_review"])

    def test_false_color_image_is_not_staged(self) -> None:
        """Мультиспектр/NDVI-раскраска не должна давать стадию."""
        import numpy as np

        rng = np.random.default_rng(0)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:, :, 0] = 0
        image[:, :, 1] = rng.integers(200, 255, size=(120, 160), dtype=np.uint8)
        image[:, :, 2] = rng.integers(0, 60, size=(120, 160), dtype=np.uint8)
        with mock.patch.object(photo_ai, "model_available", return_value=True), mock.patch.object(
            photo_ai, "detect_elements_array", side_effect=fake_detector({})
        ):
            frame = photo_ai.analyze_frame(image)
        self.assertEqual(frame["reason"], "false_color_image")
        self.assertEqual(frame["confidence"], 0.0)
        self.assertTrue(frame["needs_manual_review"])

    def test_16bit_tiff_is_convertible_for_display(self) -> None:
        """16-битный TIFF (мультиспектр дрона) не должен ронять превью.

        Регрессия: `st.image(raw_bytes)` вызывал
        `OSError: cannot write mode I;16 as JPEG`, потому что Pillow не умеет
        писать 16-битные режимы в JPEG. `to_display_rgb` приводит любой файл
        к 8-битному RGB.
        """
        import numpy as np
        from PIL import Image

        import io

        buffer = io.BytesIO()
        Image.fromarray(np.full((64, 96), 4000, dtype=np.uint16)).save(buffer, "TIFF")
        rgb = photo_ai.to_display_rgb(buffer.getvalue())
        self.assertIsNotNone(rgb)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb.shape[2], 3)

    def test_to_display_rgb_rejects_garbage(self) -> None:
        self.assertIsNone(photo_ai.to_display_rgb(b"not an image"))

    def test_no_frames_is_not_a_confident_zero(self) -> None:
        fused = photo_ai.fuse_frames([])
        self.assertEqual(fused["stage"], "0%")
        self.assertEqual(fused["confidence"], 0.0)
        self.assertTrue(fused["needs_manual_review"])

    def test_single_frame_matches_itself(self) -> None:
        fused = photo_ai.fuse_frames([self.frame("100%", 0.88)])
        self.assertEqual(fused["stage"], "100%")
        self.assertEqual(fused["frames"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

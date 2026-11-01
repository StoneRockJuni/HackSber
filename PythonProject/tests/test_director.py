"""Business-routing tests for the AI-Director."""

from __future__ import annotations

import unittest

from director import evaluate_strategy


def eye(stage: str = "50%", confidence: float = 0.9, **gate_overrides) -> dict:
    gate = {
        "image_quality_passed": True,
        "exif_gps_match": True,
        "challenge_marker_present": True,
        "error_message": None,
    }
    gate.update(gate_overrides)
    return {
        "quality_gate": gate,
        "construction_analysis": {
            "detected_stage": stage,
            "confidence_stage": confidence,
            "wall_material": "unknown",
            "confidence_material": 0.0,
            "needs_manual_review": confidence < 0.75,
            "analysis_source": "sber_yolo_onnx",
        },
        "geospatial_check": {
            "boundaries_violation": False,
            "vri_compliant": True,
            "site_development_notes": "test",
        },
    }


ROW = {
    "Наличие ПЗ": "Нет",
    "ОН зарегистрирован": "Да",
    "Залог": "Да",
    "Плановая дата": "31.12.2027",
    "Дата начала строительства": "01.01.2026",
}


class DirectorTests(unittest.TestCase):
    def test_low_ai_confidence_goes_to_manual_review(self) -> None:
        result = evaluate_strategy(eye("50%", 0.30), ROW, current_date_str="31.10.2026")
        self.assertEqual(result["verdict"]["strategy_number"], 3)
        self.assertIn("не смог уверенно", result["explanation"])

    def test_normal_50_percent_goes_to_planned_monitoring(self) -> None:
        result = evaluate_strategy(eye("50%", 0.9), ROW, current_date_str="31.10.2026")
        self.assertEqual(result["verdict"]["strategy_number"], 1)

    def test_unregistered_finished_house_goes_to_strategy_7(self) -> None:
        row = dict(ROW, **{"ОН зарегистрирован": "Нет", "Залог": "Нет"})
        result = evaluate_strategy(eye("100%", 0.9), row, current_date_str="31.10.2026")
        self.assertEqual(result["verdict"]["strategy_number"], 7)

    def test_confident_zero_is_intensive_monitoring_not_planned(self) -> None:
        """Достоверные 0 % — это риск, а не «стройка идёт по графику»."""
        result = evaluate_strategy(eye("0%", 0.95), ROW, current_date_str="31.10.2026")
        self.assertEqual(result["verdict"]["strategy_number"], 5)
        self.assertIn("не начаты", result["explanation"])

    def test_uncertain_zero_explains_why_it_does_not_block(self) -> None:
        """«Ничего не видно» != «ничего не построено»: сначала проверка."""
        result = evaluate_strategy(
            {**eye("0%", 0.55), "_meta": {"marker_source": "exif_fallback"}},
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 3)
        self.assertIn("Блокировать кредит", result["explanation"])
        self.assertIn("не над тем участком", result["explanation"])

    def test_zero_with_overdue_and_expired_goes_to_recovery(self) -> None:
        row = dict(ROW, **{"Наличие ПЗ": "Да", "Плановая дата": "01.01.2026"})
        result = evaluate_strategy(eye("0%", 0.95), row, current_date_str="31.10.2026")
        self.assertEqual(result["verdict"]["strategy_number"], 8)

    def test_proven_empty_shot_escalates_instead_of_repeat_flight(self) -> None:
        """GPS + QR подтвердили, что снят нужный участок и сейчас.

        Тогда «объекта не видно» — это факт (0 %), а не повод гонять дрон
        повторно: включается интенсивный мониторинг.
        """
        result = evaluate_strategy(
            {**eye("0%", 0.55), "_meta": {"marker_source": "qr"}},
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 5)
        self.assertEqual(result["evidence"]["level"], "strong")

    def test_unproven_empty_shot_goes_to_repeat_flight(self) -> None:
        """Без QR-маркера «не видно» может быть не тем участком — проверяем."""
        result = evaluate_strategy(
            {**eye("0%", 0.55), "_meta": {"marker_source": "exif_fallback"}},
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 3)

    def test_unrelated_file_is_blocked_even_if_quality_failed(self) -> None:
        """Чертёж/скриншот без GPS и QR уходит в СУБ, а не в «повторный вылет».

        Регрессия: раньше проверка качества стояла первой, и любой посторонний
        файл (инженерная графика, скриншот) получал стратегию 3, хотя летать
        повторно не за чем — непонятно, тот ли это объект.
        """
        result = evaluate_strategy(
            eye("0%", 0.0, image_quality_passed=False,
                exif_gps_match=False, challenge_marker_present=False),
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 6)
        self.assertEqual(result["evidence"]["level"], "none")

    def test_bad_frame_of_known_object_goes_to_repeat_flight(self) -> None:
        """Плохой кадр объекта, привязанного к заданию, — это повторный вылет."""
        result = evaluate_strategy(
            {**eye("50%", 0.9, image_quality_passed=False), "_meta": {"marker_source": "exif_fallback"}},
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 3)
        self.assertEqual(result["evidence"]["level"], "medium")

    def test_fraud_gate_has_priority_over_confidence(self) -> None:
        # Полный мусор: ни GPS, ни маркера -> фрод, даже если бы уверенность была высокой.
        result = evaluate_strategy(
            eye("0%", 0.0, exif_gps_match=False, challenge_marker_present=False),
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 6)
        self.assertEqual(result["evidence"]["level"], "none")

    def test_no_gps_but_marker_is_not_fraud(self) -> None:
        """Дрон не подписывает снимки, метаданных может не быть.

        Если маркер полётного задания есть, а GPS нет — это НЕ фрод:
        снимок уходит на ручную проверку (стратегия 3), а не в СУБ.
        """
        result = evaluate_strategy(
            eye("50%", 0.9, exif_gps_match=False),
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 3)
        self.assertEqual(result["evidence"]["level"], "weak")
        self.assertIn("не фрод", result["explanation"])

    def test_gps_without_marker_is_manual_review(self) -> None:
        result = evaluate_strategy(
            eye("100%", 0.95, challenge_marker_present=False),
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 3)
        self.assertEqual(result["evidence"]["level"], "weak")

    def test_exif_fallback_counts_as_medium_evidence(self) -> None:
        result = evaluate_strategy(
            {**eye("50%", 0.9), "_meta": {"marker_source": "exif_fallback"}},
            ROW,
            current_date_str="31.10.2026",
        )
        self.assertEqual(result["verdict"]["strategy_number"], 1)
        self.assertEqual(result["evidence"]["level"], "medium")


if __name__ == "__main__":
    unittest.main(verbosity=2)

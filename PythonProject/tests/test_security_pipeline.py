"""Regression tests for the StroyKontur security pipeline.

Run from the project root:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

The tests use only the synthetic demo images and temporary files. They do not
need a drone, a network connection, or real client data.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from security import (
    audit_chain_report,
    build_snapshot_passport,
    check_challenge_marker,
    compute_image_hash,
    extract_exif_data,
    extract_gps_and_exif,
    run_quality_gate,
    verify_audit_chain,
    verify_exif_gps_match,
    write_audit_log,
)

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo_data" / "demo_object_1_ok.jpg"
FRAUD = ROOT / "demo_data" / "demo_fraud_no_exif.jpg"
MARKER = "STROYKONTUR-DEMO-2026"


class SecurityPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="stroykontur_tests_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_hash_is_stable_and_changes_after_edit(self) -> None:
        first = compute_image_hash(DEMO)
        second = compute_image_hash(DEMO)
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[a-f0-9]{64}$")

        changed = self.tmp / "changed.jpg"
        data = bytearray(DEMO.read_bytes())
        data[-1] ^= 0x01
        changed.write_bytes(bytes(data))
        self.assertNotEqual(first, compute_image_hash(changed))

    def test_exif_gps_is_read_from_the_file(self) -> None:
        exif = extract_exif_data(DEMO)
        self.assertTrue(exif["has_exif"])
        self.assertEqual(exif["gps_source"], "exif")
        self.assertAlmostEqual(exif["gps"]["lat"], 54.8511, places=4)
        self.assertAlmostEqual(exif["gps"]["lon"], 83.1946, places=4)

        gps_info = extract_gps_and_exif(DEMO, DEMO.name)
        self.assertEqual(gps_info["source"], "exif")
        self.assertIn("Тип А", gps_info["type"])

    def test_gps_tolerance_accepts_near_and_rejects_far(self) -> None:
        self.assertTrue(verify_exif_gps_match(DEMO, (54.8511, 83.1946), tolerance_m=30))
        self.assertFalse(verify_exif_gps_match(DEMO, (55.0, 83.0), tolerance_m=30))
        self.assertFalse(verify_exif_gps_match(FRAUD, (54.8511, 83.1946), tolerance_m=30))

    def test_qr_marker_accepts_only_expected_value(self) -> None:
        self.assertTrue(check_challenge_marker(DEMO, MARKER))
        self.assertFalse(check_challenge_marker(DEMO, "WRONG-MARKER"))
        self.assertFalse(check_challenge_marker(FRAUD, MARKER))

    def test_quality_gate_happy_path_and_fraud_path(self) -> None:
        good = run_quality_gate(
            DEMO,
            filename=DEMO.name,
            expected_marker=MARKER,
            cadastral_gps=(54.8511, 83.1946),
        )
        self.assertTrue(good["image_quality_passed"])
        self.assertTrue(good["exif_gps_match"])
        self.assertTrue(good["challenge_marker_present"])
        self.assertEqual(good["_meta"]["marker_source"], "qr")

        fraud = run_quality_gate(FRAUD, filename=FRAUD.name, expected_marker=MARKER)
        self.assertTrue(fraud["image_quality_passed"])
        self.assertFalse(fraud["exif_gps_match"])
        self.assertFalse(fraud["challenge_marker_present"])
        self.assertIn("GPS", fraud["error_message"])

    def test_passport_links_hash_gps_and_custody(self) -> None:
        passport = build_snapshot_passport(
            DEMO,
            client_id="CL-TEST",
            drone_id="DRONE-TEST",
            flight_task_id="FT-TEST",
            operator_id="OP-TEST",
        )
        self.assertEqual(passport["hash"]["algorithm"], "SHA-256")
        self.assertEqual(passport["hash"]["value"], compute_image_hash(DEMO))
        self.assertEqual(passport["client_id"], "CL-TEST")
        self.assertEqual(passport["custody"][0]["step"], "capture")
        self.assertAlmostEqual(passport["gps"]["lat"], 54.8511, places=4)
        self.assertTrue(passport["verification_url"].startswith("https://"))

    def test_nested_details_are_preserved(self) -> None:
        log = self.tmp / "nested_audit.jsonl"
        record = write_audit_log(
            {
                "actor": "tester",
                "action": "decision_recorded",
                "object_hash": "e" * 64,
                "decision": "approve",
                "details": {
                    "cadastral_number": "54:19:164603:361",
                    "ai_source": "sber_yolo_onnx",
                    "ai_model_sha256": "f" * 64,
                },
            },
            str(log),
        )
        self.assertEqual(record["details"]["ai_source"], "sber_yolo_onnx")
        self.assertEqual(record["details"]["cadastral_number"], "54:19:164603:361")
        self.assertTrue(verify_audit_chain(str(log)))

    def test_audit_chain_detects_tampering(self) -> None:
        log = self.tmp / "audit.jsonl"
        write_audit_log(
            {
                "actor": "tester",
                "action": "decision_recorded",
                "object_hash": "a" * 64,
                "decision": "approve",
            },
            str(log),
        )
        write_audit_log(
            {
                "actor": "tester",
                "action": "decision_exported",
                "object_hash": "b" * 64,
                "decision": "export",
            },
            str(log),
        )
        self.assertTrue(verify_audit_chain(str(log)))
        self.assertEqual(audit_chain_report(str(log))["records"], 2)

        lines = log.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["decision"] = "reject"
        lines[0] = json.dumps(first, ensure_ascii=False, separators=(",", ":"))
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(verify_audit_chain(str(log)))
        self.assertFalse(audit_chain_report(str(log))["valid"])

    def test_audit_chain_detects_deleted_head(self) -> None:
        log = self.tmp / "audit_head.jsonl"
        for index in range(2):
            write_audit_log(
                {
                    "actor": "tester",
                    "action": f"step_{index}",
                    "object_hash": "c" * 64,
                    "decision": "ok",
                },
                str(log),
            )
        lines = log.read_text(encoding="utf-8").splitlines()
        log.write_text(lines[1] + "\n", encoding="utf-8")
        self.assertFalse(verify_audit_chain(str(log)))

    def test_corrupt_file_does_not_crash(self) -> None:
        broken = self.tmp / "broken.jpg"
        broken.write_bytes(b"this is not an image")
        result = run_quality_gate(broken, filename=broken.name)
        self.assertFalse(result["image_quality_passed"])
        self.assertIn("повреждён", result["error_message"])
        self.assertEqual(result["_meta"]["marker_source"], "none")

    def test_image_without_exif_has_no_gps(self) -> None:
        no_exif = self.tmp / "no_exif.jpg"
        with Image.open(DEMO) as image:
            image.convert("RGB").save(no_exif, "JPEG", quality=85)

        exif = extract_exif_data(no_exif)
        self.assertFalse(exif["has_exif"])
        self.assertIsNone(exif["gps"])
        gps_info = extract_gps_and_exif(no_exif, no_exif.name)
        self.assertEqual(gps_info["source"], "none")
        self.assertIsNone(gps_info["lat"])

    def test_exif_fallback_is_not_strict_proof(self) -> None:
        """A photo with EXIF but no QR must not pass the strict freshness check.

        This is the known demo weakness: `allow_exif_fallback=True` treats any
        EXIF as a freshness signal. The strict mode is the one to use when
        checking arbitrary internet photos.
        """
        no_qr_but_exif = self.tmp / "no_qr_but_exif.jpg"
        with Image.open(DEMO) as source:
            exif = source.getexif()
            image = source.convert("RGB")
        ImageDraw.Draw(image).rectangle((1180, 820, 1580, 1180), fill=(95, 140, 80))
        image.save(no_qr_but_exif, "JPEG", quality=92, exif=exif)

        strict = run_quality_gate(
            no_qr_but_exif,
            filename=no_qr_but_exif.name,
            expected_marker=MARKER,
            allow_exif_fallback=False,
        )
        demo = run_quality_gate(
            no_qr_but_exif,
            filename=no_qr_but_exif.name,
            expected_marker=MARKER,
            allow_exif_fallback=True,
        )
        self.assertFalse(strict["challenge_marker_present"])
        self.assertTrue(demo["challenge_marker_present"])
        self.assertEqual(demo["_meta"]["marker_source"], "exif_fallback")
        self.assertTrue(strict["exif_gps_match"])

    def test_audit_record_matches_json_schema(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed")

        schema_path = ROOT / "security" / "audit_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        log = self.tmp / "schema_audit.jsonl"
        record = write_audit_log(
            {
                "actor": "tester",
                "action": "decision_recorded",
                "object_hash": "d" * 64,
                "decision": "approve",
                "passport_id": "passport-demo",
            },
            str(log),
        )
        jsonschema.validate(instance=record, schema=schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)

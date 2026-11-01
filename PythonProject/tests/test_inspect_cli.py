"""Smoke test for the standalone image inspector CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo_data" / "demo_object_1_ok.jpg"
CLI = ROOT / "tools" / "inspect_image.py"


class InspectCliTests(unittest.TestCase):
    def _run(self, path: Path) -> dict:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        completed = subprocess.run(
            [sys.executable, str(CLI), "--path", str(path), "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_demo_image_reports_exif_gps_and_qr(self) -> None:
        report = self._run(DEMO)
        self.assertTrue(report["exif"]["has_exif"])
        self.assertEqual(report["gps_info"]["source"], "exif")
        self.assertAlmostEqual(report["exif"]["gps"]["lat"], 54.8511, places=4)
        self.assertTrue(report["quality_gate_strict"]["challenge_marker_present"])
        self.assertEqual(
            report["quality_gate_strict"]["_meta"]["marker_source"],
            "qr",
        )

    def test_image_without_exif_reports_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            no_exif = Path(tmp) / "no_exif.jpg"
            with Image.open(DEMO) as image:
                image.convert("RGB").save(no_exif, "JPEG", quality=85)
            report = self._run(no_exif)
            self.assertFalse(report["exif"]["has_exif"])
            self.assertEqual(report["gps_info"]["source"], "none")
            self.assertIsNone(report["exif"]["gps"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

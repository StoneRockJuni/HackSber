"""Тесты GPS-provenance: откуда именно берётся координата.

Главный вопрос на защите: «откуда вы берёте GPS?». Ответ должен быть
проверяемым, поэтому здесь проверяются все ветки:

* EXIF GPS IFD;
* XMP-пакет (DJI пишет туда координату тоже);
* имя файла (выгрузка полётного лога);
* отсутствие координат - честный `none`, а не выдуманная точка.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from security import extract_exif_data, extract_xmp_gps, gps_provenance, extract_gps_and_exif

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo_data" / "demo_object_1_ok.jpg"

XMP_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" xmlns:tiff="http://ns.adobe.com/tiff/1.0/">
   <drone-dji:GPSLatitude>{lat}</drone-dji:GPSLatitude>
   <drone-dji:GPSLongitude>{lon}</drone-dji:GPSLongitude>
   <drone-dji:GPSAltitude>{alt}</drone-dji:GPSAltitude>
   <tiff:Model>{model}</tiff:Model>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""


def insert_xmp(jpeg_bytes: bytes, lat: str, lon: str, alt: str = "120.0", model: str = "M3M") -> bytes:
    """Insert an APP1 XMP segment right after SOI, as DJI cameras do."""
    xml = XMP_TEMPLATE.format(lat=lat, lon=lon, alt=alt, model=model)
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + xml.encode("utf-8")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    assert jpeg_bytes[:2] == b"\xff\xd8", "not a JPEG"
    return jpeg_bytes[:2] + segment + jpeg_bytes[2:]


class GpsProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="gps_provenance_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _jpeg_without_metadata(self, name: str) -> Path:
        path = self.tmp / name
        with Image.open(DEMO) as image:
            image.convert("RGB").save(path, "JPEG", quality=85)
        return path

    def test_exif_gps_is_reported_as_exif_source(self) -> None:
        prov = gps_provenance(DEMO, DEMO.name)
        self.assertEqual(prov["used"]["source"], "exif")
        self.assertAlmostEqual(prov["used"]["lat"], 54.8511, places=3)
        self.assertAlmostEqual(prov["used"]["lon"], 83.1946, places=3)
        self.assertTrue(any(item["source"] == "exif" for item in prov["sources"]))
        self.assertTrue(prov["checks"], "должны быть независимые проверки тегов")
        self.assertIn("EXIF", prov["explanation"])

    def test_photo_without_metadata_has_no_gps(self) -> None:
        path = self._jpeg_without_metadata("no_meta.jpg")
        prov = gps_provenance(path, path.name)
        self.assertEqual(prov["used"]["source"], "none")
        self.assertIsNone(prov["used"]["lat"])
        self.assertEqual(prov["sources"], [])
        self.assertIn("нет", prov["explanation"].lower())

    def test_xmp_only_coordinates_are_used_as_fallback(self) -> None:
        base = self._jpeg_without_metadata("plain.jpg").read_bytes()
        path = self.tmp / "xmp_only.jpg"
        path.write_bytes(insert_xmp(base, "54.8511153", "83.1946836", model="DJI M3M"))

        xmp = extract_xmp_gps(path)
        self.assertIsNotNone(xmp, "XMP-пакет должен распознаваться")
        self.assertAlmostEqual(xmp["lat"], 54.8511153, places=5)
        self.assertAlmostEqual(xmp["lon"], 83.1946836, places=5)
        self.assertEqual(xmp["drone_model"], "DJI M3M")

        exif = extract_exif_data(path)
        self.assertEqual(exif["gps_source"], "xmp")
        prov = gps_provenance(path, path.name)
        self.assertEqual(prov["used"]["source"], "xmp")

    def test_filename_coordinates_are_marked_untrusted(self) -> None:
        path = self._jpeg_without_metadata("54.8511_83.1946.jpg")
        prov = gps_provenance(path, path.name)
        self.assertEqual(prov["used"]["source"], "flight_log")
        sources = {item["source"]: item for item in prov["sources"]}
        self.assertIn("flight_log", sources)
        self.assertFalse(sources["flight_log"]["trusted"])
        self.assertIn("имени файла", prov["explanation"])

    def test_exif_and_xmp_disagreement_is_flagged(self) -> None:
        base = DEMO.read_bytes()
        path = self.tmp / "tampered.jpg"
        # EXIF says 54.8511/83.1946, XMP says a point ~11 km away.
        path.write_bytes(insert_xmp(base, "54.9500", "83.3000"))
        prov = gps_provenance(path, path.name)
        checks = {check["name"]: check for check in prov["checks"]}
        self.assertIn("exif_xmp_agree", checks)
        self.assertFalse(checks["exif_xmp_agree"]["ok"])
        self.assertEqual(prov["used"]["source"], "exif", "EXIF имеет приоритет над XMP")

    def test_legacy_extract_gps_and_exif_keeps_contract(self) -> None:
        info = extract_gps_and_exif(DEMO, DEMO.name)
        for key in ("type", "has_exif", "lat", "lon", "date", "drone_model", "source"):
            self.assertIn(key, info)
        self.assertEqual(info["source"], "exif")
        self.assertIn("Тип А", info["type"])

    def test_demo_without_exif_returns_none_source(self) -> None:
        fraud = ROOT / "demo_data" / "demo_fraud_no_exif.jpg"
        info = extract_gps_and_exif(fraud, fraud.name)
        self.assertEqual(info["source"], "none")
        self.assertIsNone(info["lat"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

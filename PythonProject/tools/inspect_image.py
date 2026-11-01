r"""Inspect any image with the StroyKontur security pipeline.

Examples (from the project root):

    .\.venv\Scripts\python.exe tools\inspect_image.py --path C:\photos\house.jpg
    .\.venv\Scripts\python.exe tools\inspect_image.py --url https://example.com/photo.jpg
    .\.venv\Scripts\python.exe tools\inspect_image.py --path photo.jpg --lat 54.8511 --lon 83.1946
    .\.venv\Scripts\python.exe tools\inspect_image.py --url ... --marker STROYKONTUR-DEMO-2026 --passport

The tool shows:
  * where GPS came from (EXIF, file name, or nowhere);
  * raw EXIF GPS tags;
  * SHA-256;
  * quality-gate result;
  * QR/marker result;
  * optional snapshot passport.

It does not upload the image anywhere: URL download is done by this script,
and the image is processed locally.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import ExifTags, Image  # noqa: E402

from security import (  # noqa: E402
    build_snapshot_passport,
    compute_image_hash,
    extract_exif_data,
    extract_gps_and_exif,
    run_quality_gate,
)
from sber_vision import analyze_construction as analyze_with_ai  # noqa: E402
from sber_vision import model_available, model_sha256  # noqa: E402

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _raw_gps(path: Path) -> dict[str, Any]:
    """Raw GPS IFD for diagnostics, not for business logic."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            except Exception:
                raw = exif.get(34853)
                gps_ifd = raw if isinstance(raw, dict) else {}
            return {
                ExifTags.GPSTAGS.get(key, str(key)): _jsonable(value)
                for key, value in (gps_ifd or {}).items()
            }
    except Exception:
        return {}


def _download(url: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> Path:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "StroyKonturInspector/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Файл больше лимита {max_bytes // (1024 * 1024)} МБ")
        content_type = response.headers.get_content_type()
        final_url = response.geturl()

    parsed = urllib.parse.urlparse(final_url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = mimetypes.guess_extension(content_type) or ".img"
    temp_dir = Path(tempfile.mkdtemp(prefix="stroykontur_inspect_"))
    path = temp_dir / f"downloaded{suffix}"
    path.write_bytes(data)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Проверка любого изображения через security-слой СтройКонтур")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", help="Путь к локальному изображению")
    source.add_argument("--url", help="URL изображения (скачается локально во временную папку)")
    parser.add_argument("--marker", default="", help="Ожидаемый QR-маркер; пусто = принимать любой QR")
    parser.add_argument("--lat", type=float, help="Широта кадастровой точки для точной проверки")
    parser.add_argument("--lon", type=float, help="Долгота кадастровой точки для точной проверки")
    parser.add_argument("--tolerance", type=float, default=30.0, help="Допуск GPS в метрах (по умолчанию 30)")
    parser.add_argument("--passport", action="store_true", help="Построить паспорт снимка")
    parser.add_argument("--json", action="store_true", help="Вывести только JSON")
    parser.add_argument("--max-mb", type=int, default=25, help="Лимит скачивания, МБ")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if (args.lat is None) != (args.lon is None):
        parser.error("--lat и --lon нужно указывать вместе")
    if args.tolerance < 0:
        parser.error("--tolerance не может быть отрицательным")

    temp_path: Path | None = None
    try:
        if args.url:
            temp_path = _download(args.url, max_bytes=args.max_mb * 1024 * 1024)
            path = temp_path
        else:
            path = Path(args.path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        summary = extract_exif_data(path)
        gps_info = extract_gps_and_exif(path, path.name)
        cadastral = (args.lat, args.lon) if args.lat is not None and args.lon is not None else None
        gate_args = {
            "filename": path.name,
            "expected_marker": args.marker or None,
            "cadastral_gps": cadastral,
            "tolerance_m": args.tolerance,
        }
        qgate_strict = run_quality_gate(path, allow_exif_fallback=False, **gate_args)
        qgate_demo = run_quality_gate(path, allow_exif_fallback=True, **gate_args)
        digest = compute_image_hash(path)
        raw_gps = _raw_gps(path)
        ai_analysis = analyze_with_ai(path) if model_available() else None

        passport = None
        if args.passport:
            passport = build_snapshot_passport(
                path,
                client_id="CL-DEMO",
                drone_id="DRONE-DEMO",
                flight_task_id="TASK-DEMO",
                operator_id="OP-DEMO",
            )

        report = {
            "file": {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            },
            "exif": summary,
            "gps_info": gps_info,
            "raw_gps_tags": raw_gps,
            "cadastral_check": {
                "mode": "точная точка" if cadastral else "региональный fallback",
                "point": cadastral,
                "tolerance_m": args.tolerance,
            },
            "quality_gate_strict": qgate_strict,
            "quality_gate_demo_fallback": qgate_demo,
            "ai": {
                "available": model_available(),
                "model": "sberstroi_yolov8n.onnx",
                "model_sha256": model_sha256(),
                "analysis": ai_analysis,
            },
            "passport": passport,
        }

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return 0

        print("=" * 64)
        print(f"Файл:      {path}")
        print(f"Размер:    {path.stat().st_size / 1024:.1f} КБ")
        print(f"SHA-256:   {digest}")
        print(f"EXIF:      {'найден' if summary.get('has_exif') else 'не найден'}")
        print(f"GPS:       {summary.get('gps') or 'нет'}")
        print(f"Источник:  {gps_info.get('source')} ({gps_info.get('type')})")
        print(f"Дата:      {summary.get('captured_at') or 'нет'}")
        print(f"Модель:    {summary.get('drone_model') or 'нет'}")
        print("Сырые GPS-теги EXIF:")
        print(json.dumps(raw_gps, ensure_ascii=False, indent=2, default=str) if raw_gps else "  нет")
        print("-" * 64)
        print(f"Кадастровая проверка: {'точная точка' if cadastral else 'региональный fallback'}")
        print("Quality Gate (strict, без подмены EXIF):")
        print(f"  GPS match:          {qgate_strict.get('exif_gps_match')}")
        print(f"  Качество:           {qgate_strict.get('image_quality_passed')}")
        print(f"  QR/маркер:          {qgate_strict.get('challenge_marker_present')}")
        print(f"  Источник маркера:   {qgate_strict.get('_meta', {}).get('marker_source')}")
        print(f"  Payload QR:         {qgate_strict.get('_meta', {}).get('marker_payload')}")
        print(f"  Ошибки:             {qgate_strict.get('error_message') or 'нет'}")
        print("Quality Gate (demo, EXIF считается fallback-сигналом):")
        print(f"  QR/маркер:          {qgate_demo.get('challenge_marker_present')}")
        print(f"  Источник маркера:   {qgate_demo.get('_meta', {}).get('marker_source')}")
        print("-" * 64)
        if ai_analysis:
            ai_construction = ai_analysis["construction_analysis"]
            print("AI (YOLOv8n SberStroi, ONNX):")
            print(f"  Стадия:             {ai_construction.get('detected_stage')}")
            print(f"  Уверенность:        {ai_construction.get('confidence_stage')}")
            print(f"  Классы:             {ai_construction.get('detected_classes')}")
            print(f"  Ручной разбор:      {ai_construction.get('needs_manual_review')}")
            print(f"  Комментарий:        {ai_analysis['geospatial_check'].get('site_development_notes')}")
        else:
            print("AI (YOLOv8n SberStroi, ONNX): модель не найдена — используется OpenCV fallback")
        if passport:
            print("-" * 64)
            print("Паспорт снимка:")
            print(json.dumps(passport, ensure_ascii=False, indent=2, default=str))
        print("=" * 64)
        print("Примечание: соцсети и мессенджеры часто удаляют EXIF/GPS. Если GPS нет — это свойство файла, а не ошибка проверки.")
        return 0
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
                temp_path.parent.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

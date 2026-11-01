r"""Проверить ЛЮБОЕ фото: с диска, из папки-серии или по ссылке из интернета.

Показывает главное, что спрашивают на защите:

1. **Откуда взялись координаты** - EXIF GPS IFD, XMP-пакет, имя файла, или их
   нет вообще. Плюс независимые проверки тегов (WGS-84, активный фикс,
   расхождение EXIF и XMP) и честное «GPS не найден», а не выдуманная точка.
2. **Прошёл ли снимок Quality Gate** (резкость, геопривязка, QR-маркер).
3. **Что сказала нейросеть** - по каждому кадру и по серии целиком, с
   уверенностью и списком найденных элементов.
4. **Что с этим делать** - короткий вердикт для человека.

Примеры (из папки PythonProject):

    .\.venv\Scripts\python.exe tools\check_photo.py --path "C:\photos\house.jpg"
    .\.venv\Scripts\python.exe tools\check_photo.py --url "https://example.com/house.jpg"
    .\.venv\Scripts\python.exe tools\check_photo.py --dir ..\домики --overlay out.jpg
    .\.venv\Scripts\python.exe tools\check_photo.py --dir ..\домики --json

Файл никуда не отправляется: URL скачивается локально с SSRF-фильтром
(см. net_safe.py), дальше всё считается на этой машине.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net_safe import UnsafeURLError, fetch_image, NotImageError  # noqa: E402
from photo_ai import analyze_frame, fuse_frames, render_overlay, save_image  # noqa: E402
from sber_vision import CLASS_NAMES_RU, model_available, model_sha256  # noqa: E402
from security import (  # noqa: E402
    build_snapshot_passport,
    compute_image_hash,
    evidence_level,
    extract_exif_data,
    gps_provenance,
    run_quality_gate,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _load_source(args: argparse.Namespace) -> tuple[bytes, str, str]:
    """Return (bytes, name, description).

    For ``--dir`` the representative file is the first frame of the series:
    the file hash, GPS provenance and quality gate are per-file attributes,
    while the AI verdict is computed over the whole folder.
    """
    if args.url:
        fetched = fetch_image(args.url, max_bytes=args.max_mb * 1024 * 1024)
        name = Path(urllib.parse.urlsplit(fetched.final_url).path).name or "downloaded_image"
        return fetched.data, name, f"URL: {fetched.final_url}"
    if args.dir:
        folder = Path(args.dir).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise FileNotFoundError(f"в папке {folder} нет изображений")
        first = images[0]
        return (
            first.read_bytes(),
            first.name,
            f"Папка-серия: {folder} ({len(images)} кадров; паспорт и Quality Gate — по первому кадру)",
        )
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_bytes(), path.name, f"Файл: {path}"


def _human_gps(prov: dict[str, Any]) -> list[str]:
    lines = []
    used = prov["used"]
    if used["source"] == "none":
        lines.append("  GPS:            НЕ НАЙДЕН (ни EXIF, ни XMP, ни имя файла)")
    else:
        lines.append(
            f"  GPS:            {used['lat']}, {used['lon']}"
            + (f" (высота {used['alt_m']} м)" if used.get("alt_m") is not None else "")
            + f"  <- источник: {used['source']}"
        )
    for source in prov["sources"]:
        lines.append(
            f"    - {source['source']:11s} {source['lat']}, {source['lon']}  {source['label']}"
        )
    for check in prov["checks"]:
        mark = "OK  " if check["ok"] else "!!  "
        lines.append(f"    {mark} {check['name']}: {check['detail']}")
    return lines


def _gate_lines(gate: dict[str, Any], has_cadastral: bool) -> list[str]:
    meta = gate.get("_meta", {})
    mode = "точная кадастровая точка" if has_cadastral else "региональный fallback (Томская обл.)"
    return [
        f"  Резкость:       {meta.get('sharpness')} (норма > 120), яркость {meta.get('brightness')}",
        f"  GPS совпадает:  {gate.get('exif_gps_match')}  [{mode}]",
        f"  QR/маркер:      {gate.get('challenge_marker_present')} (источник: {meta.get('marker_source')}, payload: {meta.get('marker_payload')})",
        f"  Ошибки:         {gate.get('error_message') or 'нет'}",
    ]


def _verdict(fused: dict[str, Any], gate: dict[str, Any], prov: dict[str, Any], evidence: dict[str, Any] | None = None) -> str:
    evidence = evidence or {}
    if not gate.get("image_quality_passed"):
        return "Снимок не прошёл контроль качества: нужен повторный кадр, а не решение по нему."
    if evidence.get("level") == "none":
        return (
            "Снимок не к чему привязать: нет ни GPS, ни QR-маркера полётного задания. "
            "Это риск фрода — автоматика блокируется, материал уходит в СУБ."
        )
    if evidence.get("level") == "weak":
        return (
            f"Снимок доказан частично ({evidence.get('reason')}). Это не фрод, но и не "
            "основание для авторешения: нужен повторный кадр с маркером или инспектор."
        )
    if prov["used"]["source"] == "none" and gate.get("exif_gps_match") is False:
        return (
            "Геопривязки нет: файл без GPS. Для банка это «не подтверждено», "
            "а не «объект не построен». Нужен оригинал с EXIF или QR-маркер полётного задания."
        )
    if fused["needs_manual_review"]:
        return (
            f"Автоматическое решение не принимается (уверенность {fused['confidence']:.2f} < 0.75): "
            "повторный облёт или ручной разбор инспектора."
        )
    stage = fused["stage"]
    if stage == "100%":
        return "Объект закрыт контуром (готовая кровля): можно фиксировать 100% и переходить к оформлению/залогу."
    if stage == "50%":
        return "Контур не закрыт: фундамент/стены без готовой кровли — стадия 50%."
    return "Элементов дома не найдено: похоже на 0%, но требуется подтверждение (повторный кадр/осмотр)."


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    data, name, description = _load_source(args)
    cadastral = (args.lat, args.lon) if args.lat is not None and args.lon is not None else None

    gate_args = {
        "filename": name,
        "expected_marker": args.marker or None,
        "cadastral_gps": cadastral,
        "tolerance_m": args.tolerance,
    }
    gate_strict = run_quality_gate(data, allow_exif_fallback=False, **gate_args)
    gate_demo = run_quality_gate(data, allow_exif_fallback=True, **gate_args)
    prov = gps_provenance(data, name)

    payloads: list[Any] = []
    if model_available() and not args.no_ai:
        if args.dir:
            for path in sorted(Path(args.dir).expanduser().resolve().iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    payloads.append(path)
        else:
            payloads.append(data)

    frames = []
    used_payloads = []
    for payload in payloads:
        frame = analyze_frame(payload)
        if frame:
            frames.append(frame)
            used_payloads.append(payload)
    fused = fuse_frames(frames) if frames else None

    best_index = 0
    if frames:
        best_index = max(
            range(len(frames)),
            key=lambda i: (frames[i]["stage"] == (fused or {}).get("stage"), frames[i]["confidence"]),
        )

    report: dict[str, Any] = {
        "description": description,
        "file": {
            "name": name,
            "size_bytes": len(data),
            "sha256": compute_image_hash(data),
        },
        "gps_provenance": prov,
        "evidence": evidence_level(gate_strict, gate_strict.get("_meta")),
        "exif": extract_exif_data(data),
        "quality_gate_strict": gate_strict,
        "quality_gate_demo": {
            "challenge_marker_present": gate_demo.get("challenge_marker_present"),
            "marker_source": gate_demo.get("_meta", {}).get("marker_source"),
        },
        "ai": {
            "available": model_available(),
            "model_sha256": model_sha256(),
            "frames_analyzed": len(frames),
            "per_frame": [
                {
                    "stage": frame["stage"],
                    "confidence": frame["confidence"],
                    "reason": frame["reason"],
                    "classes": frame["classes"],
                    "cues": frame["cues"],
                }
                for frame in frames
            ],
            "fused": fused,
        },
    }
    if args.passport:
        report["passport"] = build_snapshot_passport(
            data,
            client_id="CL-DEMO",
            drone_id="DRONE-DEMO",
            flight_task_id="TASK-DEMO",
            operator_id="OP-DEMO",
        )

    if args.overlay and frames:
        image = render_overlay(
            used_payloads[best_index],
            frames[best_index],
            stage=fused["stage"] if fused else None,
            confidence=fused["confidence"] if fused else None,
        )
        if image is not None:
            save_image(args.overlay, image)
            report["overlay_path"] = str(Path(args.overlay).resolve())
    return report


def _print_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    print("=" * 72)
    print(report["description"])
    print(f"  Размер:         {report['file']['size_bytes'] / 1024:.1f} КБ")
    print(f"  SHA-256:        {report['file']['sha256']}")
    print("-" * 72)
    print("1. ОТКУДА GPS")
    for line in _human_gps(report["gps_provenance"]):
        print(line)
    print(f"  Пояснение:      {report['gps_provenance']['explanation']}")
    print("-" * 72)
    print("2. КАЧЕСТВО И ПОДЛИННОСТЬ (strict: QR обязателен)")
    for line in _gate_lines(report["quality_gate_strict"], bool(args.lat is not None)):
        print(line)
    evidence = report.get("evidence") or {}
    print(f"  Доказанность:   {evidence.get('level')} — {evidence.get('reason')}")
    print("-" * 72)
    print("3. НЕЙРОСЕТЬ (YOLOv8n SberStroi, ONNX, локально)")
    ai = report["ai"]
    if not ai["available"]:
        print("  Модель не установлена - доступен только OpenCV fallback.")
    elif not ai["frames_analyzed"]:
        print("  Кадры не проанализированы.")
    else:
        fused = ai["fused"]
        print(
            f"  Кадров:         {ai['frames_analyzed']}  согласованность {fused['agreement']:.0%}"
        )
        print(f"  Стадия:         {fused['stage']}  (уверенность {fused['confidence']:.2f})")
        print(f"  Причина:        {fused['reason']}")
        counts = fused.get("evidence", {})
        print(
            "  Голоса:         "
            f"100% x{counts.get('frames_100', 0)}, 50% x{counts.get('frames_50', 0)}, "
            f"0% x{counts.get('frames_0', 0)}"
        )
        for index, frame in enumerate(ai["per_frame"], start=1):
            classes = ", ".join(CLASS_NAMES_RU.get(c, c) for c in frame["classes"]) or "-"
            print(
                f"    кадр {index}: {frame['stage']:4s} {frame['confidence']:.2f} "
                f"[{frame['reason']}] элементы: {classes}"
            )
    print("-" * 72)
    print("4. ЧТО ДЕЛАТЬ")
    fused = ai.get("fused") or {"stage": "0%", "confidence": 0.0, "needs_manual_review": True}
    gate = report["quality_gate_strict"]
    print("  " + _verdict(fused, gate, report["gps_provenance"], report.get("evidence")))
    if report.get("overlay_path"):
        print(f"  Разметка сохранена: {report['overlay_path']}")
    print("=" * 72)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Проверка любого фото: GPS-происхождение + Quality Gate + нейросеть"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", help="Путь к изображению")
    source.add_argument("--url", help="Прямая ссылка на изображение (http/https)")
    source.add_argument("--dir", help="Папка с серией кадров одного объекта")
    parser.add_argument("--marker", default="", help="Ожидаемый QR-маркер; пусто = любой QR")
    parser.add_argument("--lat", type=float, help="Широта кадастровой точки")
    parser.add_argument("--lon", type=float, help="Долгота кадастровой точки")
    parser.add_argument("--tolerance", type=float, default=30.0, help="Допуск GPS, метры")
    parser.add_argument("--max-mb", type=int, default=25, help="Лимит скачивания по URL, МБ")
    parser.add_argument("--overlay", help="Куда сохранить размеченный кадр (JPG/PNG)")
    parser.add_argument("--passport", action="store_true", help="Построить паспорт снимка")
    parser.add_argument("--no-ai", action="store_true", help="Не запускать нейросеть")
    parser.add_argument("--json", action="store_true", help="Вывести только JSON")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat и --lon нужно указывать вместе")
    if args.tolerance < 0:
        parser.error("--tolerance не может быть отрицательным")
    try:
        report = _build_report(args)
    except (UnsafeURLError, NotImageError) as exc:
        print(f"ОТКАЗАНО: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Файл не найден: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_report(report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

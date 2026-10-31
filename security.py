#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
security.py — модуль ИБ платформы «СтройКонтур».

Реализовано:
  1. SHA-256 снимка (доказательство неизменности файла).
  2. Проверка GPS из EXIF (или GPS-сайдкара для комплектации «только GPS»)
     на соответствие координатам кадастрового участка.
  3. Проверка челлендж-маркера (QR через OpenCV либо demo-сайдкар).
  4. «Паспорт снимка» — JSON с происхождением и ссылкой верификации.
  5. Append-only Audit Trail с цепочкой hash_prev -> hash_self.
  6. Проверка целостности всей цепочки аудита.

Зависимости:
  - стандартная библиотека Python;
  - ОПЦИОНАЛЬНО: opencv-python-headless (cv2) для чтения QR с изображения.
    Без cv2 модуль работает, но QR-проверка переходит в demo-режим
    через сайдкар <image>.marker.json / <image>.marker.txt.

Секреты НЕ хранятся в коде. Для усиления аудита можно задать:
  AUDIT_HMAC_KEY            — ключ HMAC-SHA256 (>= 32 байт), либо
  AUDIT_HMAC_KEY_FILE       — путь к файлу с ключом.
  PASSPORT_HMAC_KEY         — ключ подписи паспорта, либо
  PASSPORT_HMAC_KEY_FILE    — путь к файлу с ключом.
Ключи в целевом контуре должны жить в KMS/HSM, а не в переменных окружения
прикладного процесса. Переменные окружения здесь — только для хакатона.

Ограничение: hash-chain делает журнал tamper-EVIDENT, но не tamper-PROOF.
Чтобы нельзя было удалить последние записи или пересчитать всю цепочку,
в production нужен внешний анкор: WORM-хранилище, KMS/HSM, ежедневная
публикация head-hash в отдельном контуре банка.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import mimetypes
import os
import re
import struct
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

try:  # OpenCV нужен только для реального распознавания QR.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - зависит от окружения
    cv2 = None

try:  # Межпроцессная блокировка журнала на Windows.
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover
    msvcrt = None

try:  # Межпроцессная блокировка журнала на POSIX.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None


GENESIS_HASH = "0" * 64

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+\-]{0,127}$")
_ACTION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:\-]{0,63}$")
_DECISION_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

_ALLOWED_AUDIT_ROLES = frozenset({
    "inspector",
    "risk_officer",
    "auditor",
    "ai_director",
    "contractor",
    "system",
    "security_admin",
    "service",
})

_ALLOWED_EVENT_KEYS = frozenset({
    "timestamp",          # принимается только как claimed_timestamp в details
    "actor",
    "actor_role",
    "action",
    "object_hash",
    "decision",
    "details",
    "correlation_id",
})

_REQUIRED_AUDIT_FIELDS = frozenset({
    "seq",
    "timestamp",
    "actor",
    "actor_role",
    "action",
    "object_hash",
    "decision",
    "details",
    "hash_prev",
    "hash_self",
})

_HMAC_FIELDS = frozenset({"hmac_alg", "hmac_sha256"})
_ALLOWED_AUDIT_FIELDS = frozenset(set(_REQUIRED_AUDIT_FIELDS) | set(_HMAC_FIELDS) | {"correlation_id"})
_THREAD_LOCK = threading.Lock()

_MAX_EXIF_BLOB = 4 * 1024 * 1024
_MAX_JPEG_SCAN = 64 * 1024 * 1024
_MAX_SIDECAR_BYTES = 1024 * 1024
_MAX_TIFF_VALUE_BYTES = 1024 * 1024
_MAX_DETAILS_BYTES = 32 * 1024


# ---------------------------------------------------------------------------
# Базовые криптографические и JSON-утилиты
# ---------------------------------------------------------------------------

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_hex(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _canonical_json(value: Any) -> bytes:
    """Детерминированное JSON-представление — основа всех хэшей."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Объект несериализуем в канонический JSON") from exc
    return text.encode("utf-8")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _load_secret(env_name: str, file_env_name: str, *, min_bytes: int = 32) -> bytes | None:
    """Ключ из env/файла. В коде нет ни одного секрета по умолчанию."""
    raw = os.getenv(env_name)
    if raw:
        data = raw.encode("utf-8")
        source = env_name
    else:
        file_name = os.getenv(file_env_name)
        if not file_name:
            return None
        path = Path(file_name)
        if not path.is_file():
            raise FileNotFoundError(f"Файл ключа не найден: {file_env_name}")
        if path.stat().st_size > 4096:
            raise ValueError(f"Файл ключа слишком большой: {file_env_name}")
        data = path.read_bytes().strip()
        source = file_env_name

    if len(data) < min_bytes:
        raise ValueError(
            f"{source}: ключ короче {min_bytes} байт. "
            "Для HMAC-SHA256 используйте минимум 32 случайных байта."
        )
    return data


def _validate_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} должен быть строкой")
    value = value.strip()
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field}: допустимы латиница, цифры и символы _ . : @ + - "
            "(1..128 символов, начинается с буквы или цифры)"
        )
    return value


def _validate_timestamp(value: Any, field: str = "timestamp") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} должен быть непустой ISO-8601 строкой")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field}: некорректный ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field}: обязателен часовой пояс (например, Z)")
    return value


def _require_str(event: Mapping[str, Any], field: str) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"event.{field} обязателен и должен быть непустой строкой")
    return value.strip()


def _coerce_coordinates(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{field} должен быть кортежем/списком (lat, lon)")
    try:
        lat = float(value[0])
        lon = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: координаты должны быть числами") from exc
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError(f"{field}: координаты должны быть конечными числами")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"{field}: широта вне диапазона [-90; 90]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"{field}: долгота вне диапазона [-180; 180]")
    return lat, lon


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по большому кругу, метры (WGS-84, средний радиус Земли)."""
    radius = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


# ---------------------------------------------------------------------------
# 1. Хэш снимка
# ---------------------------------------------------------------------------

def compute_image_hash(path: str) -> str:
    """
    SHA-256 файла снимка.

    Хэш считается по байтам «как есть»: любое изменение пикселя, EXIF,
    пересохранение JPEG или добавление байта меняет значение.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Снимок не найден: {image_path}")

    digest = hashlib.sha256()
    with image_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 2. EXIF / GPS без внешних API
# ---------------------------------------------------------------------------
# Pillow не требуется: разбираем JPEG APP1/TIFF/PNG eXIf вручную.
# Это важно для on-premise-контура и для случая, когда на сервере нет
# тяжёлых зависимостей. Все смещения проверяются, размеры ограничены.

_TIFF_TYPE_SIZE = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    6: 1,   # SBYTE
    7: 1,   # UNDEFINED
    8: 2,   # SSHORT
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
    13: 4,  # IFD
}


def _read_exif_blob(path: Path) -> bytes | None:
    """Возвращает TIFF-блок EXIF (без префикса Exif\\0\\0) или None."""
    with path.open("rb") as fh:
        head = fh.read(12)
        if not head:
            return None

        # TIFF-файл (в датасете встречаются multispectral .TIF DJI).
        if head.startswith(b"II*\x00") or head.startswith(b"MM\x00*"):
            fh.seek(0)
            return fh.read(_MAX_EXIF_BLOB)

        # PNG: чанк eXIf.
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            fh.seek(8)
            while fh.tell() < _MAX_JPEG_SCAN:
                chunk_header = fh.read(8)
                if len(chunk_header) != 8:
                    return None
                length = int.from_bytes(chunk_header[:4], "big")
                chunk_type = chunk_header[4:8]
                if length > _MAX_EXIF_BLOB:
                    return None
                if chunk_type == b"eXIf":
                    return fh.read(length)
                if chunk_type == b"IEND":
                    return None
                fh.seek(length + 4, 1)  # data + CRC32
            return None

        if not head.startswith(b"\xff\xd8"):
            return None

        # JPEG: идём по маркерам до Start Of Scan, ищем APP1/Exif.
        fh.seek(2)
        while fh.tell() < _MAX_JPEG_SCAN:
            byte = fh.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker_byte = fh.read(1)
            while marker_byte == b"\xff":  # fill bytes
                marker_byte = fh.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]

            # Маркеры без длины сегмента.
            if marker in (0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
                          0xD7, 0xD8, 0xD9):
                if marker == 0xD9:
                    break
                continue
            if marker == 0xDA:  # SOS — дальше сжатые данные
                break

            length_bytes = fh.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = int.from_bytes(length_bytes, "big") - 2
            if segment_length < 0 or segment_length > _MAX_JPEG_SCAN:
                break

            if marker == 0xE1 and segment_length >= 6:
                data = fh.read(segment_length)
                if data.startswith(b"Exif\x00\x00"):
                    return data[6:]
            else:
                fh.seek(segment_length, 1)
    return None


def _read_ifd(tiff: bytes, offset: int, byte_order: str) -> tuple[dict[int, tuple[int, int, bytes]], int]:
    """Читает IFD: {tag: (type, count, raw_bytes)} и смещение следующего IFD."""
    if offset < 8 or offset + 2 > len(tiff):
        raise ValueError("Некорректное смещение IFD")
    entry_count = struct.unpack_from(byte_order + "H", tiff, offset)[0]
    if entry_count > 512:
        raise ValueError("Слишком много записей IFD")
    entries: dict[int, tuple[int, int, bytes]] = {}
    position = offset + 2
    for _ in range(entry_count):
        if position + 12 > len(tiff):
            raise ValueError("Выход за границы IFD")
        tag, type_id, count = struct.unpack_from(byte_order + "HHI", tiff, position)
        value_field = tiff[position + 8:position + 12]
        unit = _TIFF_TYPE_SIZE.get(type_id)
        if unit is None or count > 1_000_000:
            position += 12
            continue
        value_size = unit * count
        if value_size > _MAX_TIFF_VALUE_BYTES:
            position += 12
            continue
        if value_size <= 4:
            value_bytes = value_field[:value_size]
        else:
            pointer = struct.unpack(byte_order + "I", value_field)[0]
            if pointer + value_size > len(tiff):
                position += 12
                continue
            value_bytes = tiff[pointer:pointer + value_size]
        entries[tag] = (type_id, count, value_bytes)
        position += 12
    next_offset = 0
    if position + 4 <= len(tiff):
        next_offset = struct.unpack_from(byte_order + "I", tiff, position)[0]
    return entries, next_offset


def _decode_tiff_value(entry: tuple[int, int, bytes], byte_order: str) -> Any:
    type_id, count, data = entry
    if count == 0:
        return b"" if type_id in (1, 6, 7) else ("" if type_id == 2 else [])
    try:
        if type_id == 2:
            return data.rstrip(b"\x00").decode("utf-8", errors="replace")
        if type_id in (1, 7):
            return list(data[:count])
        if type_id == 6:
            return list(struct.unpack(byte_order + "b" * count, data[:count]))
        if type_id == 3:
            return list(struct.unpack(byte_order + "H" * count, data[:2 * count]))
        if type_id == 8:
            return list(struct.unpack(byte_order + "h" * count, data[:2 * count]))
        if type_id in (4, 9, 13):
            fmt = "I" if type_id in (4, 13) else "i"
            return list(struct.unpack(byte_order + fmt * count, data[:4 * count]))
        if type_id in (5, 10):
            fmt = "II" if type_id == 5 else "ii"
            values = struct.unpack(byte_order + fmt * count, data[:8 * count])
            return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
        if type_id == 11:
            return list(struct.unpack(byte_order + "f" * count, data[:4 * count]))
        if type_id == 12:
            return list(struct.unpack(byte_order + "d" * count, data[:8 * count]))
    except (struct.error, UnicodeError):
        return None
    return None


def _first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _rational_to_float(value: Any) -> float | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = value
        if denominator == 0:
            return None
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _dms_to_decimal(values: Any, reference: Any) -> float | None:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    parts: list[float] = []
    for item in values[:3]:
        number = _rational_to_float(item)
        if number is None:
            return None
        parts.append(number)
    decimal = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    ref = str(reference or "").strip().upper()
    if ref in {"S", "W"}:
        decimal = -decimal
    return decimal


def _parse_exif_datetime(value: Any, offset: Any = None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            iso = parsed.strftime("%Y-%m-%dT%H:%M:%S")
            offset_text = str(offset or "").strip()
            if re.fullmatch(r"[+-]\d{2}:\d{2}", offset_text):
                return iso + offset_text
            return iso
        except ValueError:
            continue
    return None


def _parse_gps_timestamp(date_value: Any, time_value: Any) -> str | None:
    date_text = str(date_value or "").strip()
    if not re.fullmatch(r"\d{4}:\d{2}:\d{2}", date_text):
        return None
    if not isinstance(time_value, (list, tuple)) or len(time_value) < 3:
        return None
    parts = [_rational_to_float(item) for item in time_value[:3]]
    if any(part is None for part in parts):
        return None
    year, month, day = (int(part) for part in date_text.split(":"))
    hour, minute, second = parts  # type: ignore[misc]
    try:
        return (
            datetime(year, month, day, int(hour), int(minute), int(second),
                     tzinfo=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except ValueError:
        return None


def _extract_exif(path: Path) -> dict[str, Any]:
    """
    Возвращает нормализованные метаданные:
    {present, gps, captured_at, captured_at_utc, make, model, ...}.
    Ошибка разбора не «роняет» загрузку: снимок уходит на ручную проверку.
    """
    result: dict[str, Any] = {
        "present": False,
        "gps": None,
        "captured_at": None,
        "captured_at_utc": None,
        "captured_at_source": None,
        "make": None,
        "model": None,
        "datetime_original": None,
        "warning": None,
    }
    try:
        tiff = _read_exif_blob(path)
        if not tiff or len(tiff) < 8:
            return result
        if tiff.startswith(b"II"):
            byte_order = "<"
        elif tiff.startswith(b"MM"):
            byte_order = ">"
        else:
            return result
        magic = struct.unpack_from(byte_order + "H", tiff, 2)[0]
        if magic != 42:
            return result
        ifd0_offset = struct.unpack_from(byte_order + "I", tiff, 4)[0]
        ifd0, _ = _read_ifd(tiff, ifd0_offset, byte_order)
        result["present"] = True

        make_raw = ifd0.get(0x010F)
        model_raw = ifd0.get(0x0110)
        if make_raw:
            result["make"] = _decode_tiff_value(make_raw, byte_order)
        if model_raw:
            result["model"] = _decode_tiff_value(model_raw, byte_order)

        exif_ifd: dict[int, tuple[int, int, bytes]] = {}
        exif_pointer = ifd0.get(0x8769)
        if exif_pointer:
            pointer = _first_value(_decode_tiff_value(exif_pointer, byte_order))
            if isinstance(pointer, int) and 0 < pointer < len(tiff):
                exif_ifd, _ = _read_ifd(tiff, pointer, byte_order)

        datetime_entry = exif_ifd.get(0x9003) or ifd0.get(0x0132)
        offset_entry = exif_ifd.get(0x9011)
        datetime_text = (
            _decode_tiff_value(datetime_entry, byte_order) if datetime_entry else None
        )
        offset_text = (
            _decode_tiff_value(offset_entry, byte_order) if offset_entry else None
        )
        result["datetime_original"] = datetime_text
        result["captured_at"] = _parse_exif_datetime(datetime_text, offset_text)
        if result["captured_at"]:
            result["captured_at_source"] = "exif_datetime_original"

        gps_pointer = ifd0.get(0x8825)
        if gps_pointer:
            pointer = _first_value(_decode_tiff_value(gps_pointer, byte_order))
            if isinstance(pointer, int) and 0 < pointer < len(tiff):
                gps_ifd, _ = _read_ifd(tiff, pointer, byte_order)
                lat_entry = gps_ifd.get(0x0002)
                lon_entry = gps_ifd.get(0x0004)
                lat_ref = gps_ifd.get(0x0001)
                lon_ref = gps_ifd.get(0x0003)
                if lat_entry and lon_entry:
                    latitude = _dms_to_decimal(
                        _decode_tiff_value(lat_entry, byte_order),
                        _decode_tiff_value(lat_ref, byte_order) if lat_ref else "",
                    )
                    longitude = _dms_to_decimal(
                        _decode_tiff_value(lon_entry, byte_order),
                        _decode_tiff_value(lon_ref, byte_order) if lon_ref else "",
                    )
                    if (
                        latitude is not None
                        and longitude is not None
                        and -90.0 <= latitude <= 90.0
                        and -180.0 <= longitude <= 180.0
                    ):
                        altitude = None
                        altitude_entry = gps_ifd.get(0x0006)
                        if altitude_entry:
                            altitude = _rational_to_float(
                                _first_value(_decode_tiff_value(altitude_entry, byte_order))
                            )
                        altitude_ref = None
                        altitude_ref_entry = gps_ifd.get(0x0005)
                        if altitude_ref_entry:
                            altitude_ref = _first_value(
                                _decode_tiff_value(altitude_ref_entry, byte_order)
                            )
                        if altitude is not None and altitude_ref == 1:
                            altitude = -altitude

                        gps_date_entry = gps_ifd.get(0x001D)
                        gps_time_entry = gps_ifd.get(0x0007)
                        gps_utc = None
                        if gps_date_entry and gps_time_entry:
                            gps_utc = _parse_gps_timestamp(
                                _decode_tiff_value(gps_date_entry, byte_order),
                                _decode_tiff_value(gps_time_entry, byte_order),
                            )
                        result["gps"] = {
                            "lat": round(latitude, 7),
                            "lon": round(longitude, 7),
                            "altitude_m": round(altitude, 2) if altitude is not None else None,
                            "accuracy_m": None,
                            "source": "exif_gps",
                            "source_file": path.name,
                            "captured_at": gps_utc,
                        }
                        if gps_utc:
                            result["captured_at_utc"] = gps_utc
                            if not result["captured_at"]:
                                result["captured_at"] = gps_utc
                                result["captured_at_source"] = "exif_gps_timestamp"
    except (OSError, ValueError, struct.error, UnicodeError) as exc:
        result["warning"] = f"EXIF не разобран: {type(exc).__name__}"
        result["gps"] = None
    return result


def _sidecar_candidates(image_path: Path, suffix: str) -> list[Path]:
    """Варианты имени сайдкара: photo.JPG.gps.json, photo.gps.json и т.п."""
    candidates = [
        image_path.with_name(image_path.name + suffix),
        image_path.with_suffix(suffix),
        image_path.with_name(image_path.stem + suffix),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _read_json_limited(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size > _MAX_SIDECAR_BYTES:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _ci_get(mapping: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _extract_gps_from_mapping(data: Mapping[str, Any]) -> dict[str, Any] | None:
    nested = data.get("gps")
    if nested is None:
        nested = data.get("location")
    if nested is None:
        nested = data.get("telemetry")
    if nested is None:
        nested = data

    latitude: Any = None
    longitude: Any = None
    if isinstance(nested, (list, tuple)) and len(nested) >= 2:
        latitude, longitude = nested[0], nested[1]
    elif isinstance(nested, Mapping):
        latitude = _ci_get(nested, "lat", "latitude")
        longitude = _ci_get(nested, "lon", "lng", "longitude")
    elif isinstance(nested, str):
        parts = [part.strip() for part in nested.split(",")]
        if len(parts) == 2:
            latitude, longitude = parts[0], parts[1]

    if latitude is None or longitude is None:
        return None
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    accuracy = _ci_get(data, "accuracy_m", "accuracy") if isinstance(data, Mapping) else None
    try:
        accuracy_value = float(accuracy) if accuracy is not None else None
    except (TypeError, ValueError):
        accuracy_value = None
    captured_at = _ci_get(data, "captured_at", "timestamp", "datetime", "date_time")
    source = _ci_get(data, "source")
    return {
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "altitude_m": None,
        "accuracy_m": round(accuracy_value, 2) if accuracy_value is not None else None,
        "source": f"sidecar:{source}" if source else "sidecar_gps",
        "source_file": None,
        "captured_at": str(captured_at) if captured_at else None,
    }


def _read_sidecar_gps(image_path: Path) -> dict[str, Any] | None:
    """GPS-only комплектация: координаты из подписанного/принятого телеметрийного JSON."""
    for suffix in (".gps.json", ".telemetry.json", ".meta.json"):
        for candidate in _sidecar_candidates(image_path, suffix):
            try:
                data = _read_json_limited(candidate)
                if not data:
                    continue
                gps = _extract_gps_from_mapping(data)
                if gps:
                    gps["source_file"] = candidate.name
                    return gps
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
                continue
    return None


def _analyze_image(image_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    exif = _extract_exif(image_path)
    gps = exif.get("gps")
    if gps is None:
        gps = _read_sidecar_gps(image_path)
    return exif, gps


def verify_exif_gps_match(
    image_path: str,
    cadastral_gps: tuple,
    tolerance_m: float = 30.0,
) -> bool:
    """
    Сверяет GPS снимка с координатами кадастрового участка.

    Логика для двух комплектаций из ТЗ:
      * EXIF + GPS  -> берём GPS из EXIF;
      * только GPS  -> если EXIF GPS нет, читаем <image>.gps.json /
                       <image>.telemetry.json (телеметрия подрядчика).

    Возвращает False при отсутствии GPS, битых координатах или выходе
    расстояния за tolerance_m. Исключения — только на некорректные аргументы.
    """
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Снимок не найден: {image}")
    cadastral_lat, cadastral_lon = _coerce_coordinates(cadastral_gps, "cadastral_gps")
    try:
        tolerance = float(tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance_m должен быть числом") from exc
    if not math.isfinite(tolerance) or not (0.0 <= tolerance <= 100_000.0):
        raise ValueError("tolerance_m должен быть в диапазоне [0; 100000]")

    _, gps = _analyze_image(image)
    if not gps:
        return False
    try:
        distance = _haversine_m(
            float(gps["lat"]), float(gps["lon"]), cadastral_lat, cadastral_lon
        )
    except (KeyError, TypeError, ValueError):
        return False
    return distance <= tolerance


# ---------------------------------------------------------------------------
# 3. Челлендж-маркер
# ---------------------------------------------------------------------------

def _read_marker_sidecar(image_path: Path) -> tuple[str | None, str | None]:
    for suffix in (".marker.json", ".challenge.json"):
        for candidate in _sidecar_candidates(image_path, suffix):
            try:
                data = _read_json_limited(candidate)
                if not data:
                    continue
                marker = _ci_get(data, "marker", "challenge", "token", "payload")
                if marker is not None:
                    return str(marker).strip(), "sidecar_json"
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
                continue
    for suffix in (".marker.txt", ".challenge.txt"):
        for candidate in _sidecar_candidates(image_path, suffix):
            try:
                if candidate.is_file() and candidate.stat().st_size <= 4096:
                    return candidate.read_text(encoding="utf-8").strip(), "sidecar_text"
            except (OSError, UnicodeError):
                continue
    return None, None


def _marker_matches(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    return hmac.compare_digest(
        actual.strip().encode("utf-8"), expected.strip().encode("utf-8")
    )


def _check_challenge_marker(image_path: str, expected_marker: str) -> tuple[bool, str]:
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Снимок не найден: {image}")
    if not isinstance(expected_marker, str):
        raise TypeError("expected_marker должен быть строкой")
    expected = expected_marker.strip()
    if not expected or len(expected.encode("utf-8")) > 2048:
        raise ValueError("expected_marker должен быть непустым (до 2048 байт)")

    # 3.1. Реальный QR: cv2.QRCodeDetector. Работает офлайн, без внешних API.
    if cv2 is not None:
        try:
            picture = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if picture is not None:
                detector = cv2.QRCodeDetector()
                decoded_values: list[str] = []
                if hasattr(detector, "detectAndDecodeMulti"):
                    ok, decoded, _points, _straight = detector.detectAndDecodeMulti(picture)
                    if ok and decoded is not None:
                        decoded_values = [str(item) for item in decoded if item]
                else:  # pragma: no cover - старые версии OpenCV
                    value, _points, _straight = detector.detectAndDecode(picture)
                    if value:
                        decoded_values = [str(value)]
                if any(_marker_matches(value, expected) for value in decoded_values):
                    return True, "qr_cv2"
                # Если OpenCV установлен, сайдкар НЕ используется как обход
                # визуальной проверки: нет QR — нет подтверждения.
                return False, "qr_not_found"
            return False, "qr_decode_error"
        except Exception:  # cv2.error и ошибки декодирования не должны ломать поток
            return False, "qr_decode_error"

    # 3.2. Demo-режим без OpenCV: маркер приходит сайдкаром. В production
    # этот путь должен быть выключен; он нужен только для ноутбука жюри.
    actual, method = _read_marker_sidecar(image)
    if _marker_matches(actual, expected):
        return True, method or "sidecar"
    return False, "not_found"


def check_challenge_marker(image_path: str, expected_marker: str) -> bool:
    """
    Проверяет наличие ожидаемого QR/цветового маркера на снимке.

    В хакатонном прототипе:
      * если установлен cv2 — декодирует QR и сравнивает payload;
      * иначе — проверяет <image>.marker.json / <image>.marker.txt.
    Ожидаемый маркер должен быть одноразовым (TTL 15–30 минут) и выдаваться
    на конкретное полётное задание. Сам по себе этот флаг не доказывает
    подлинность; он работает только вместе с hash + audit trail + подписью.
    """
    return _check_challenge_marker(image_path, expected_marker)[0]


# ---------------------------------------------------------------------------
# 4. Паспорт снимка
# ---------------------------------------------------------------------------

def _verification_base_url() -> str:
    base = os.getenv("VERIFY_BASE_URL", "https://verify.stroykontur.local/p").strip()
    if base.startswith("https://"):
        return base.rstrip("/")
    if base.startswith("http://") and os.getenv("ALLOW_INSECURE_VERIFY_URL") == "1":
        return base.rstrip("/")
    raise ValueError(
        "VERIFY_BASE_URL должен начинаться с https:// "
        "(http:// разрешён только при ALLOW_INSECURE_VERIFY_URL=1)"
    )


def _passport_digest(passport: Mapping[str, Any]) -> str:
    """Хэш паспорта без самого поля passport_sha256 и подписи."""
    clone = json.loads(json.dumps(passport, ensure_ascii=False))
    integrity = clone.get("integrity")
    if isinstance(integrity, dict):
        for key in ("passport_sha256", "signature_alg", "signature"):
            integrity.pop(key, None)
    return _sha256_hex(_canonical_json(clone))


def build_snapshot_passport(
    image_path: str,
    client_id: str,
    drone_id: str,
    flight_task_id: str,
    *,
    cadastral_gps: tuple | None = None,
    tolerance_m: float = 30.0,
    expected_marker: str | None = None,
) -> dict:
    """
    Создаёт JSON «Паспорт снимка».

    Обязательные поля ТЗ: hash, время съёмки, GPS, drone_id, flight_task_id,
    кто снимал, ссылка верификации (QR payload). Дополнительно фиксируются
    источник GPS/времени, результат cadastral-check, маркер и data quality.

    ВАЖНО: client_id — псевдоним (UUID), ФИО/адрес/номер договора сюда НЕ
    попадают. Полный путь к файлу тоже не сохраняется — только имя файла.
    """
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Снимок не найден: {image}")

    client_id = _validate_identifier(client_id, "client_id")
    drone_id = _validate_identifier(drone_id, "drone_id")
    flight_task_id = _validate_identifier(flight_task_id, "flight_task_id")
    operator_id = (os.getenv("SNAPSHOT_OPERATOR_ID") or "").strip() or "unattested"
    operator_id = _validate_identifier(operator_id, "SNAPSHOT_OPERATOR_ID")

    snapshot_hash = compute_image_hash(str(image))
    exif, gps = _analyze_image(image)
    if gps is not None:
        for key in ("lat", "lon"):
            if not isinstance(gps.get(key), (int, float)):
                gps = None
                break

    captured_at = exif.get("captured_at")
    captured_at_source = exif.get("captured_at_source")
    if not captured_at and gps:
        captured_at = gps.get("captured_at")
        captured_at_source = "gps_telemetry" if captured_at else None

    if expected_marker is None:
        expected_marker = os.getenv("CHALLENGE_EXPECTED_MARKER")
    marker_status = "not_checked"
    marker_method = "none"
    marker_hash = None
    if expected_marker:
        expected_marker = str(expected_marker).strip()
        if expected_marker:
            marker_hash = _sha256_hex(expected_marker.encode("utf-8"))
            marker_ok, marker_method = _check_challenge_marker(str(image), expected_marker)
            marker_status = "verified" if marker_ok else "failed"

    cadastral_check: dict[str, Any]
    if cadastral_gps is None:
        cadastral_check = {
            "checked": False,
            "matched": None,
            "reason": "cadastral_gps_not_provided",
        }
    else:
        cadastral_lat, cadastral_lon = _coerce_coordinates(cadastral_gps, "cadastral_gps")
        try:
            tolerance = float(tolerance_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("tolerance_m должен быть числом") from exc
        if not math.isfinite(tolerance) or not (0.0 <= tolerance <= 100_000.0):
            raise ValueError("tolerance_m должен быть в диапазоне [0; 100000]")
        if gps:
            distance = _haversine_m(
                float(gps["lat"]), float(gps["lon"]), cadastral_lat, cadastral_lon
            )
            cadastral_check = {
                "checked": True,
                "matched": distance <= tolerance,
                "distance_m": round(distance, 2),
                "tolerance_m": tolerance,
            }
        else:
            cadastral_check = {
                "checked": False,
                "matched": None,
                "reason": "gps_not_available",
                "tolerance_m": tolerance,
            }

    warnings: list[str] = []
    if not exif.get("present"):
        warnings.append("EXIF отсутствует или не разобран")
    if not gps:
        warnings.append("GPS отсутствует: снимок нельзя привязать к участку")
    elif str(gps.get("source", "")).startswith("sidecar"):
        warnings.append("GPS взят из сайдкара телеметрии, а не из EXIF")
    if not captured_at:
        warnings.append("Время съёмки не подтверждено метаданными")
    if marker_status == "failed":
        warnings.append("Челлендж-маркер не найден или не совпал")
    elif marker_status == "not_checked":
        warnings.append("Челлендж-маркер не проверялся")
    if cadastral_check.get("checked") and not cadastral_check.get("matched"):
        warnings.append("GPS не совпал с кадастровым участком в допуске")
    if operator_id == "unattested":
        warnings.append("Личность оператора не подтверждена (SNAPSHOT_OPERATOR_ID)")

    trust_level = "unverified"
    if (
        exif.get("present")
        and exif.get("gps")
        and marker_status == "verified"
        and cadastral_check.get("matched") is True
    ):
        trust_level = "high"
    elif gps and marker_status == "verified":
        trust_level = "medium"
    elif gps:
        trust_level = "low"

    issued_at = _utc_now()
    passport_seed = "|".join(
        [snapshot_hash, client_id, drone_id, flight_task_id, str(captured_at or "")]
    )
    passport_id = "sp_" + _sha256_hex(passport_seed.encode("utf-8"))[:32]
    verification_url = (
        _verification_base_url() + "/" + quote(passport_id, safe="")
    )

    gps_for_passport = None
    if gps:
        gps_for_passport = {
            "lat": round(float(gps["lat"]), 7),
            "lon": round(float(gps["lon"]), 7),
            "altitude_m": gps.get("altitude_m"),
            "accuracy_m": gps.get("accuracy_m"),
            "source": gps.get("source"),
            "source_file": gps.get("source_file"),
            "captured_at": gps.get("captured_at"),
        }

    passport: dict[str, Any] = {
        "schema_version": "1.0",
        "passport_id": passport_id,
        "issued_at": issued_at,
        "verification": {
            "url": verification_url,
            "qr_payload": verification_url,
            "method": "passport_lookup_by_id",
            "pii_in_qr": False,
            "note": "QR содержит только идентификатор паспорта; ПДн в QR не кодируются.",
        },
        "snapshot": {
            "sha256": snapshot_hash,
            "file_name": image.name,
            "size_bytes": image.stat().st_size,
            "mime_type": mimetypes.guess_type(image.name)[0] or "application/octet-stream",
            "captured_at": captured_at,
            "captured_at_utc": exif.get("captured_at_utc"),
            "captured_at_source": captured_at_source,
        },
        "location": {
            "gps": gps_for_passport,
            "cadastral_check": cadastral_check,
        },
        "challenge_marker": {
            "status": marker_status,
            "method": marker_method,
            "expected_marker_sha256": marker_hash,
            "ttl_policy": "one-time marker per flight task, TTL 15-30 min",
        },
        "flight": {
            "drone_id": drone_id,
            "flight_task_id": flight_task_id,
            "operator_id": operator_id,
        },
        "client": {
            "client_id": client_id,
            "pseudonymized": True,
            "pii_access": "restricted",
        },
        "source_metadata": {
            "exif_present": bool(exif.get("present")),
            "camera_make": exif.get("make"),
            "camera_model": exif.get("model"),
            "datetime_original": exif.get("datetime_original"),
            "gps_source": gps.get("source") if gps else None,
            "exif_warning": exif.get("warning"),
        },
        "data_quality": {
            "trust_level": trust_level,
            "gps_available": bool(gps),
            "captured_at_available": bool(captured_at),
            "warnings": warnings,
        },
        "chain_of_custody": {
            "status": "registered",
            "events": [
                {
                    "at": issued_at,
                    "actor": operator_id,
                    "action": "SNAPSHOT_REGISTERED",
                    "object_sha256": snapshot_hash,
                    "source": "contractor_upload",
                }
            ],
        },
        "legal": {
            "framework": "152-ФЗ",
            "client_data": "pseudonymized",
            "storage": "on-premise",
            "retention": "по политике банка/договору",
        },
        "integrity": {
            "snapshot_sha256": snapshot_hash,
            "passport_sha256": None,
        },
    }

    passport["integrity"]["passport_sha256"] = _passport_digest(passport)
    signing_key = _load_secret("PASSPORT_HMAC_KEY", "PASSPORT_HMAC_KEY_FILE")
    if signing_key:
        passport["integrity"]["signature_alg"] = "HMAC-SHA256"
        passport["integrity"]["signature"] = _hmac_hex(
            signing_key, _canonical_json(passport)
        )
    else:
        passport["integrity"]["signature_alg"] = "none"
        passport["integrity"]["signature"] = None
    return passport


# ---------------------------------------------------------------------------
# 5. Append-only Audit Trail
# ---------------------------------------------------------------------------

def _prepare_log_path(log_path: str) -> Path:
    path = Path(log_path)
    if path.exists() and path.is_symlink():
        raise ValueError("Audit log не может быть символической ссылкой")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _audit_lock(path: Path):
    """Одна запись за раз: thread lock + OS file lock."""
    _THREAD_LOCK.acquire()
    lock_path = Path(str(path) + ".lock")
    if lock_path.exists() and lock_path.is_symlink():
        _THREAD_LOCK.release()
        raise ValueError("Lock-файл не может быть символической ссылкой")

    fd: int | None = None
    locked = False
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.chmod(lock_path, 0o600)
        except OSError:
            pass
        if msvcrt is not None:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\x00")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        elif fcntl is not None:  # pragma: no cover - POSIX
            fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if fd is not None:
            try:
                if locked:
                    if msvcrt is not None:
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    elif fcntl is not None:  # pragma: no cover - POSIX
                        fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        _THREAD_LOCK.release()


def _read_last_line(path: Path) -> bytes | None:
    """Последняя строка журнала без загрузки всего файла в память."""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        if size == 0:
            return None
        fh.seek(max(0, size - 4))
        tail = fh.read()
        if not tail.endswith(b"\n"):
            raise ValueError("Audit log обрывается без завершающего перевода строки")
        if tail.endswith(b"\n\n") or tail.endswith(b"\r\n\r\n"):
            raise ValueError("Audit log содержит пустую строку в конце")

        position = size
        buffer = b""
        while position > 0 and len(buffer) < 1_048_576:
            step = min(262_144, position)
            position -= step
            fh.seek(position)
            buffer = fh.read(step) + buffer
            if buffer.count(b"\n") >= 2:
                break
        lines = [line for line in buffer.split(b"\n") if line]
        if not lines:
            return None
        return lines[-1]


def _record_hash_body(record: Mapping[str, Any]) -> dict[str, Any]:
    """Поля, покрытые hash_self: всё, кроме hash_self и HMAC-полей."""
    return {
        key: value
        for key, value in record.items()
        if key not in _HMAC_FIELDS and key != "hash_self"
    }


def _record_hmac_body(record: Mapping[str, Any]) -> dict[str, Any]:
    """Поля, покрытые HMAC: всё, включая hash_self, кроме HMAC-полей."""
    return {key: value for key, value in record.items() if key not in _HMAC_FIELDS}


def _validate_audit_record(
    record: Mapping[str, Any],
    *,
    signing_key: bytes | None,
    require_hmac: bool,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("Запись аудита должна быть JSON-объектом")
    missing = _REQUIRED_AUDIT_FIELDS - set(record)
    if missing:
        raise ValueError(f"В записи аудита нет полей: {sorted(missing)}")
    unknown = set(record) - _ALLOWED_AUDIT_FIELDS
    if unknown:
        raise ValueError(f"В записи аудита неизвестные поля: {sorted(unknown)}")

    seq = record.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ValueError("seq должен быть целым числом >= 1")
    _validate_timestamp(record.get("timestamp"))

    actor = record.get("actor")
    if not isinstance(actor, str) or not _IDENTIFIER_RE.fullmatch(actor):
        raise ValueError("actor имеет недопустимый формат")
    role = record.get("actor_role")
    if not isinstance(role, str) or not _ROLE_RE.fullmatch(role) or role not in _ALLOWED_AUDIT_ROLES:
        raise ValueError("actor_role имеет недопустимый формат/значение")
    action = record.get("action")
    if not isinstance(action, str) or not _ACTION_RE.fullmatch(action):
        raise ValueError("action имеет недопустимый формат")
    object_hash = record.get("object_hash")
    if not isinstance(object_hash, str) or not _SHA256_RE.fullmatch(object_hash):
        raise ValueError("object_hash должен быть SHA-256 в нижнем регистре")
    decision = record.get("decision")
    if not isinstance(decision, str) or not _DECISION_RE.fullmatch(decision):
        raise ValueError("decision имеет недопустимый формат")
    details = record.get("details")
    if not isinstance(details, Mapping):
        raise ValueError("details должен быть JSON-объектом")
    if len(_canonical_json(details)) > _MAX_DETAILS_BYTES:
        raise ValueError("details превышает лимит 32 КБ")
    if "correlation_id" in record:
        _validate_identifier(record.get("correlation_id"), "correlation_id")

    hash_prev = record.get("hash_prev")
    hash_self = record.get("hash_self")
    if not isinstance(hash_prev, str) or not _SHA256_RE.fullmatch(hash_prev):
        raise ValueError("hash_prev должен быть SHA-256 в нижнем регистре")
    if not isinstance(hash_self, str) or not _SHA256_RE.fullmatch(hash_self):
        raise ValueError("hash_self должен быть SHA-256 в нижнем регистре")

    expected_self = _sha256_hex(_canonical_json(_record_hash_body(record)))
    if not hmac.compare_digest(expected_self, hash_self):
        raise ValueError("hash_self не соответствует содержимому записи")

    has_hmac = "hmac_sha256" in record
    if has_hmac:
        if record.get("hmac_alg") != "HMAC-SHA256":
            raise ValueError("hmac_alg должен быть HMAC-SHA256")
        signature = record.get("hmac_sha256")
        if not isinstance(signature, str) or not _SHA256_RE.fullmatch(signature):
            raise ValueError("hmac_sha256 имеет недопустимый формат")
        if signing_key is None:
            raise ValueError("Запись подписана HMAC, но ключ не настроен")
        expected_hmac = _hmac_hex(signing_key, _canonical_json(_record_hmac_body(record)))
        if not hmac.compare_digest(expected_hmac, signature):
            raise ValueError("HMAC записи не совпал")
    else:
        if "hmac_alg" in record:
            raise ValueError("hmac_alg без hmac_sha256 недопустим")
        if require_hmac:
            raise ValueError("Ожидалась HMAC-подпись записи, но её нет")


def _read_last_audit_record(path: Path, signing_key: bytes | None) -> dict[str, Any] | None:
    line = _read_last_line(path)
    if line is None:
        return None
    record = json.loads(line.decode("utf-8"))
    _validate_audit_record(
        record,
        signing_key=signing_key,
        require_hmac=signing_key is not None,
    )
    return record


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("Не удалось записать audit log")
        view = view[written:]


def write_audit_log(event: dict, log_path: str = "audit_trail.jsonl") -> dict:
    """
    Добавляет событие в Audit Trail (JSONL, append-only).

    Гарантии:
      * timestamp ставит сервер (UTC), а не клиент — нельзя «состарить» событие;
      * hash_prev связывает запись с предыдущей;
      * hash_self покрывает всю запись, включая hash_prev;
      * при наличии AUDIT_HMAC_KEY запись дополнительно подписывается HMAC;
      * запись идёт под межпроцессной блокировкой и fsync.

    Event: {actor, actor_role?, action, object_hash, decision, details?,
            correlation_id?, timestamp?}. Клиентский timestamp сохраняется
    только как details.claimed_timestamp и не влияет на цепочку.
    """
    if not isinstance(event, Mapping):
        raise TypeError("event должен быть объектом (dict)")
    unknown = set(event) - _ALLOWED_EVENT_KEYS
    if unknown:
        raise ValueError(
            f"Неизвестные поля события: {sorted(unknown)}. "
            "Дополните context кладите в event.details."
        )

    actor = _validate_identifier(_require_str(event, "actor"), "actor")
    action = _require_str(event, "action")
    if not _ACTION_RE.fullmatch(action):
        raise ValueError("action: допустимы латиница, цифры и _ . : - (до 64 символов)")
    object_hash = _require_str(event, "object_hash").lower()
    if not _SHA256_RE.fullmatch(object_hash):
        raise ValueError(
            "object_hash должен быть SHA-256 (64 hex). "
            "Для произвольного объекта хэшируйте его канонический JSON."
        )
    decision = _require_str(event, "decision")
    if not _DECISION_RE.fullmatch(decision):
        raise ValueError("decision: допустимы латиница, цифры и _ . : - (до 64 символов)")

    role = str(event.get("actor_role") or os.getenv("AUDIT_ACTOR_ROLE", "system")).strip()
    if not _ROLE_RE.fullmatch(role) or role not in _ALLOWED_AUDIT_ROLES:
        raise ValueError(f"actor_role вне допустимого списка: {sorted(_ALLOWED_AUDIT_ROLES)}")

    details_value = event.get("details") or {}
    if not isinstance(details_value, Mapping):
        raise ValueError("details должен быть JSON-объектом")
    details = dict(details_value)
    if "timestamp" in event and event["timestamp"] is not None:
        details.setdefault("claimed_timestamp", str(event["timestamp"]))
    if len(_canonical_json(details)) > _MAX_DETAILS_BYTES:
        raise ValueError("details превышает лимит 32 КБ")

    correlation_id = None
    if event.get("correlation_id") is not None:
        correlation_id = _validate_identifier(
            event["correlation_id"], "correlation_id"
        )

    path = _prepare_log_path(log_path)
    signing_key = _load_secret("AUDIT_HMAC_KEY", "AUDIT_HMAC_KEY_FILE")

    with _audit_lock(path):
        previous = _read_last_audit_record(path, signing_key) if path.exists() else None
        if previous is None:
            sequence = 1
            hash_prev = GENESIS_HASH
        else:
            sequence = previous["seq"] + 1
            hash_prev = previous["hash_self"]

        body: dict[str, Any] = {
            "seq": sequence,
            "timestamp": _utc_now(),
            "actor": actor,
            "actor_role": role,
            "action": action,
            "object_hash": object_hash,
            "decision": decision,
            "details": details,
            "hash_prev": hash_prev,
        }
        if correlation_id:
            body["correlation_id"] = correlation_id
        body["hash_self"] = _sha256_hex(_canonical_json(body))

        record = dict(body)
        if signing_key is not None:
            record["hmac_alg"] = "HMAC-SHA256"
            record["hmac_sha256"] = _hmac_hex(signing_key, _canonical_json(body))

        payload = _canonical_json(record) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags, 0o600)
        try:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
    return record


def verify_audit_chain(log_path: str = "audit_trail.jsonl") -> bool:
    """
    Проверяет всю цепочку Audit Trail.

    Проверяется:
      * каждая строка — валидный JSON и валидная запись по схеме;
      * hash_self пересчитывается из содержимого;
      * hash_prev равен hash_self предыдущей записи;
      * seq идёт без пропусков с 1;
      * HMAC, если журнал подписан;
      * опционально — внешний анкор AUDIT_EXPECTED_HEAD / AUDIT_EXPECTED_COUNT.

    Возврат False при любом нарушении. Отсутствующий файл — тоже False:
    для банковского аудита «журнала нет» не равно «журнал цел».
    Удаление последних записей без внешнего анкора не детектируется —
    поэтому AUDIT_EXPECTED_HEAD/COUNT нужно ежедневно фиксировать в WORM.
    """
    try:
        path = Path(log_path)
        if not path.is_file() or path.is_symlink():
            return False
        if path.stat().st_size == 0:
            expected_count = os.getenv("AUDIT_EXPECTED_COUNT")
            expected_head = os.getenv("AUDIT_EXPECTED_HEAD")
            if expected_count and int(expected_count) != 0:
                return False
            if expected_head and expected_head != GENESIS_HASH:
                return False
            return True

        with path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                return False

        signing_key = _load_secret("AUDIT_HMAC_KEY", "AUDIT_HMAC_KEY_FILE")
        previous_hash = GENESIS_HASH
        expected_seq = 1
        line_count = 0
        last_hash = GENESIS_HASH

        with path.open("r", encoding="utf-8", newline="") as fh:
            for raw_line in fh:
                raw_line = raw_line.rstrip("\r\n")
                if not raw_line:
                    return False
                record = json.loads(raw_line)
                _validate_audit_record(
                    record,
                    signing_key=signing_key,
                    require_hmac=signing_key is not None,
                )
                if record["seq"] != expected_seq:
                    return False
                if not hmac.compare_digest(record["hash_prev"], previous_hash):
                    return False
                previous_hash = record["hash_self"]
                last_hash = record["hash_self"]
                line_count += 1
                expected_seq += 1

        expected_count = os.getenv("AUDIT_EXPECTED_COUNT")
        if expected_count is not None and int(expected_count) != line_count:
            return False
        expected_head = os.getenv("AUDIT_EXPECTED_HEAD")
        if expected_head is not None and not hmac.compare_digest(expected_head, last_hash):
            return False
        return True
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return False


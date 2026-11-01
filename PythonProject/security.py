"""Security layer for the StroyKontur hackathon prototype.

The module intentionally has no external API calls and no hard-coded secrets.
Configuration is read from environment variables:

    PASSPORT_VERIFY_BASE_URL  - public verification URL prefix (optional)
    PASSPORT_SIGNING_KEY      - HMAC key for passport signatures (optional)
    AUDIT_LOG_PATH            - default audit log path (optional)

What is implemented:
    * SHA-256 image hashing;
    * EXIF GPS extraction and distance check against a cadastral point;
    * QR / colour challenge-marker check;
    * "snapshot passport" JSON with chain-of-custody fields;
    * append-only JSONL audit log with a SHA-256 hash chain;
    * audit-chain verification;
    * a compatibility quality gate used by the Streamlit UI.

Residual risks are documented in security/threat_model.md.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import io
import json
import math
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import cv2
import numpy as np
from PIL import ExifTags, Image

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

AUDIT_GENESIS = "0" * 64
AUDIT_SCHEMA_VERSION = "1.0"
PASSPORT_SCHEMA_VERSION = "1.0"

_AUDIT_LOCK = threading.Lock()
_REQUIRED_AUDIT_FIELDS = ("actor", "action", "object_hash", "decision")
_JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe"}
_PNG_EXTENSIONS = {".png"}


class AuditChainError(RuntimeError):
    """Raised when the existing audit log is corrupted and cannot be extended."""


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    """Stable byte representation used for hashes and signatures."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _read_source_bytes(source: Any) -> bytes:
    """Read a path, bytes object or file-like object without consuming it."""
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read"):
        position = None
        try:
            position = source.tell()
        except Exception:
            pass
        try:
            source.seek(0)
            data = source.read()
        finally:
            if position is not None:
                try:
                    source.seek(position)
                except Exception:
                    pass
        if isinstance(data, str):
            data = data.encode("utf-8")
        return bytes(data)
    raise TypeError(f"Unsupported image source: {type(source)!r}")


def _source_name(source: Any) -> str:
    if isinstance(source, (str, os.PathLike)):
        return Path(source).name
    name = getattr(source, "name", None)
    return str(name) if name else "uploaded_image"


def _guess_content_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in _JPEG_EXTENSIONS:
        return "image/jpeg"
    if suffix in _PNG_EXTENSIONS:
        return "image/png"
    return "application/octet-stream"


def _decode_image(source: Any) -> np.ndarray | None:
    raw = _read_source_bytes(source)
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            numerator, denominator = value
            return float(numerator) / float(denominator) if float(denominator) else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_degrees(dms: Any, ref: Any = None) -> float | None:
    """Convert EXIF degrees/minutes/seconds to decimal degrees."""
    if not dms:
        return None
    try:
        if len(dms) != 3:
            return None
        degrees = _as_float(dms[0])
        minutes = _as_float(dms[1])
        seconds = _as_float(dms[2])
        if degrees is None or minutes is None or seconds is None:
            return None
        result = degrees + minutes / 60.0 + seconds / 3600.0
        ref_text = ref.decode("ascii", "ignore") if isinstance(ref, bytes) else str(ref or "")
        if ref_text.strip().upper() in {"S", "W"}:
            result = -result
        return result
    except (TypeError, ValueError):
        return None


def _parse_exif_datetime(raw: Any) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    text = str(raw).strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            parsed = dt.datetime.strptime(text, fmt)
            return parsed.isoformat(timespec="seconds"), text
        except ValueError:
            continue
    return None, text or None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _coerce_gps(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = re.split(r"[;,]", value)
        if len(parts) < 2:
            return None
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return None
    try:
        if len(value) < 2:
            return None
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# EXIF / GPS
# ---------------------------------------------------------------------------


def extract_exif_data(source: Any) -> dict[str, Any]:
    """Return a small, safe EXIF summary for an image.

    The function never raises for a normal "no EXIF" image; it returns empty
    fields instead. This is important because the bank receives both Type A
    (EXIF + GPS) and Type B (GPS only) packages.
    """
    data = _read_source_bytes(source)
    result: dict[str, Any] = {
        "has_exif": False,
        "captured_at": None,
        "captured_at_raw": None,
        "gps": None,
        "gps_source": None,
        "drone_model": None,
        "camera_make": None,
    }
    try:
        with Image.open(io.BytesIO(data)) as image:
            exif = image.getexif()
            if not exif:
                # No EXIF at all, but the file may still carry an XMP packet.
                _apply_xmp_fallback(result, data)
                return result

            result["has_exif"] = True
            make = exif.get(271)  # Make
            model = exif.get(272)  # Model
            result["camera_make"] = str(make).strip() if make else None
            result["drone_model"] = str(model).strip() if model else None

            captured_raw = exif.get(36867) or exif.get(36868) or exif.get(306)
            captured_at, captured_at_raw = _parse_exif_datetime(captured_raw)
            result["captured_at"] = captured_at
            result["captured_at_raw"] = captured_at_raw

            gps_ifd: dict[Any, Any] = {}
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)  # Pillow >= 9
            except Exception:
                try:
                    gps_ifd = exif.get_ifd(0x8825)
                except Exception:
                    raw_gps = exif.get(34853)
                    if isinstance(raw_gps, dict):
                        gps_ifd = raw_gps

            gps = {ExifTags.GPSTAGS.get(key, str(key)): value for key, value in (gps_ifd or {}).items()}
            lat = _dms_to_degrees(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
            lon = _dms_to_degrees(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                alt = _as_float(gps.get("GPSAltitude"))
                if gps.get("GPSAltitudeRef") in (1, b"\x01"):
                    alt = -alt if alt is not None else None
                result["gps"] = {
                    "lat": round(lat, 7),
                    "lon": round(lon, 7),
                    "alt_m": round(alt, 2) if alt is not None else None,
                }
                result["gps_source"] = "exif"

            # Fallback: DJI writes its coordinate into the XMP packet too, and
            # that packet survives some editors that strip the EXIF GPS IFD.
            if result["gps"] is None:
                _apply_xmp_fallback(result, data)
    except Exception:
        # Corrupted EXIF must not crash the upload pipeline. The caller sees
        # has_exif=False and can route the image to manual review.
        return result
    return result


def _apply_xmp_fallback(result: dict[str, Any], data: bytes) -> None:
    """Fill GPS/camera fields from XMP when the EXIF GPS IFD is empty."""
    xmp = extract_xmp_gps(data)
    if not xmp:
        return
    result["gps"] = {
        "lat": xmp["lat"],
        "lon": xmp["lon"],
        "alt_m": xmp.get("alt_m"),
    }
    result["gps_source"] = "xmp"
    if not result.get("drone_model") and xmp.get("drone_model"):
        result["drone_model"] = xmp["drone_model"]
    if not result.get("camera_make") and xmp.get("camera_make"):
        result["camera_make"] = xmp["camera_make"]


def _format_exif_date(value: str | None) -> str:
    if not value:
        return "Из полетного задания"
    try:
        return dt.datetime.fromisoformat(value).strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# GPS provenance: where exactly did the coordinate come from?
# ---------------------------------------------------------------------------
#
# The bank asks one question over and over: "откуда вы берёте GPS?". The
# honest answer is a list of *sources inside the file*, in priority order:
#
#   1. EXIF GPS IFD of the JPEG/TIFF itself (the drone writes it on capture);
#   2. XMP packet in APP1 (DJI writes drone-dji:GPSLatitude there as well, and
#      it survives some editors that strip the EXIF GPS IFD);
#   3. coordinates encoded in the file name, e.g. 54.8511_83.1946.jpg, which
#      is how the contractor's flight log exports frames;
#   4. nothing at all.
#
# No IP geolocation, no map matching, no device location, no LLM guessing.

_XMP_WHEN = re.compile(r'(?:exif:DateTimeOriginal|xmp:CreateDate|photoshop:DateCreated)\s*=\s*"([^"]{4,40})"')
_XMP_MODEL = re.compile(r'tiff:Model\s*=\s*"([^"]{1,80})"')
_XMP_MAKE = re.compile(r'tiff:Make\s*=\s*"([^"]{1,80})"')

GPS_SOURCE_PRIORITY = ("exif", "xmp", "flight_log")


def _xmp_lookup(text: str, names: Sequence[str]) -> tuple[str, str] | None:
    """Read an XMP field written either as an attribute or as an element.

    DJI uses attributes (``drone-dji:GPSLatitude="54.8"``), but hand-written or
    converted XMP often uses elements (``<drone-dji:GPSLatitude>54.8<...``),
    so both forms are supported.
    """
    for name in names:
        match = re.search(
            r"([A-Za-z0-9_.:-]*" + re.escape(name) + r')\s*=\s*"([^"]{1,80})"', text
        )
        if match:
            return match.group(2), match.group(1)
        match = re.search(
            r"<([A-Za-z0-9_.:-]*" + re.escape(name) + r")\s*>\s*([^<]{1,80}?)\s*</", text
        )
        if match:
            return match.group(2), match.group(1)
    return None


def _xmp_blocks(raw: bytes) -> list[str]:
    """Extract XMP packets from JPEG/PNG bytes without extra dependencies."""
    blocks: list[str] = []
    cursor = 0
    while True:
        start = raw.find(b"<x:xmpmeta", cursor)
        if start < 0:
            break
        end = raw.find(b"</x:xmpmeta>", start)
        if end < 0:
            blocks.append(raw[start : start + 262_144].decode("utf-8", "ignore"))
            break
        blocks.append(raw[start : end + 12].decode("utf-8", "ignore"))
        cursor = end + 12
        if len(blocks) >= 4:
            break
    return blocks


def _parse_xmp_coordinate(text: str, negative_ref: str) -> float | None:
    """Parse ``54.8511``, ``54,51.0669N`` and ``54,51,4.0152N`` forms."""
    value = (text or "").strip().strip('"')
    if not value:
        return None
    hemisphere = ""
    if value[-1:].upper() in {"N", "S", "E", "W"}:
        hemisphere = value[-1:].upper()
        value = value[:-1].strip()
    value = value.replace(" D", "").replace("°", " ").strip()
    parts = [part for part in re.split(r"[,\s]+", value) if part]
    try:
        numbers = [float(part.replace(",", ".")) for part in parts[:3]]
    except ValueError:
        return None
    if not numbers:
        return None
    result = numbers[0]
    if len(numbers) >= 2:
        result += numbers[1] / 60.0
    if len(numbers) >= 3:
        result += numbers[2] / 3600.0
    if hemisphere and hemisphere in negative_ref:
        result = -result
    return result


def extract_xmp_gps(source: Any) -> dict[str, Any] | None:
    """Read GPS and capture metadata from an XMP packet, if one exists.

    Returns ``None`` when the file has no XMP or no usable coordinate.
    """
    raw = _read_source_bytes(source)
    packets = _xmp_blocks(raw)
    if not packets:
        return None
    text = "\n".join(packets)

    lat_raw = _xmp_lookup(text, ("GPSLatitude", "GpsLatitude", "GPSLat"))
    lon_raw = _xmp_lookup(text, ("GPSLongitude", "GpsLongitude", "GPSLon"))
    if not lat_raw or not lon_raw:
        return None

    lat = _parse_xmp_coordinate(lat_raw[0], "S")
    lon = _parse_xmp_coordinate(lon_raw[0], "W")
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None

    altitude = None
    alt_raw = _xmp_lookup(text, ("GPSAltitude", "GpsAltitude", "RelativeAltitude"))
    if alt_raw:
        try:
            altitude = float(str(alt_raw[0]).replace(",", "."))
        except ValueError:
            altitude = None

    when = _xmp_lookup(text, ("DateTimeOriginal", "CreateDate", "DateCreated"))
    model = _xmp_lookup(text, ("Model",))
    make = _xmp_lookup(text, ("Make",))
    return {
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "alt_m": round(altitude, 2) if altitude is not None else None,
        "keys": [lat_raw[1], lon_raw[1]],
        "captured_at_raw": when[0] if when else None,
        "camera_make": make[0] if make else None,
        "drone_model": model[0] if model else None,
    }


def _filename_gps(filename: str) -> tuple[float, float] | None:
    match = re.search(r"(\d{2}\.\d{3,})[_-](\d{2}\.\d{3,})", filename or "")
    if not match:
        return None
    lat, lon = float(match.group(1)), float(match.group(2))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


def _gps_checks(exif_gps: dict[str, Any] | None, xmp: dict[str, Any] | None, name_gps: tuple[float, float] | None,
                raw_tags: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Independent sanity checks that make GPS provenance explainable."""
    checks: list[dict[str, Any]] = []
    tags = raw_tags or {}

    if exif_gps:
        lat, lon = float(exif_gps["lat"]), float(exif_gps["lon"])
        checks.append(
            {
                "name": "coordinates_in_range",
                "ok": -90 <= lat <= 90 and -180 <= lon <= 180,
                "detail": f"lat={lat}, lon={lon}",
            }
        )
        checks.append(
            {
                "name": "not_null_island",
                "ok": abs(lat) > 0.001 or abs(lon) > 0.001,
                "detail": "0,0 обычно означает, что GPS-модуль не получил фикс",
            }
        )
        datum = str(tags.get("GPSMapDatum") or "").strip("\x00 ")
        checks.append(
            {
                "name": "wgs84_datum",
                "ok": not datum or "WGS" in datum.upper(),
                "detail": datum or "тег GPSMapDatum отсутствует",
            }
        )
        status = str(tags.get("GPSStatus") or "").strip("\x00 ")
        checks.append(
            {
                "name": "gps_fix_active",
                "ok": status.upper() in {"", "A"},
                "detail": status or "тег GPSStatus отсутствует",
            }
        )
        checks.append(
            {
                "name": "altitude_present",
                "ok": exif_gps.get("alt_m") is not None,
                "detail": f"{exif_gps.get('alt_m')} м",
            }
        )

    if exif_gps and xmp:
        distance = _haversine_m(exif_gps["lat"], exif_gps["lon"], xmp["lat"], xmp["lon"])
        checks.append(
            {
                "name": "exif_xmp_agree",
                "ok": distance <= 50.0,
                "detail": f"расхождение EXIF и XMP: {distance:.1f} м",
            }
        )
    if exif_gps and name_gps:
        distance = _haversine_m(exif_gps["lat"], exif_gps["lon"], name_gps[0], name_gps[1])
        checks.append(
            {
                "name": "exif_filename_agree",
                "ok": distance <= 200.0,
                "detail": f"расхождение EXIF и имени файла: {distance:.1f} м",
            }
        )
    return checks


def gps_provenance(image: Any, filename: str = "") -> dict[str, Any]:
    """Explain exactly where a coordinate came from, and how far it can be trusted.

    Returns a dict with:

    * ``used`` - the source actually used downstream (or ``none``);
    * ``sources`` - every candidate found in the file, with raw values;
    * ``checks`` - independent sanity checks on the metadata;
    * ``explanation`` - one sentence for a human/auditor.

    This function never guesses. If the file has no coordinate, the answer is
    ``none`` - which is the normal, expected result for a photo downloaded
    from a social network or a messenger, because they strip metadata.
    """
    exif = extract_exif_data(image)
    # ``extract_exif_data`` already falls back to XMP, so only trust its `gps`
    # as an EXIF source when it says so; otherwise the XMP branch below adds it.
    exif_gps = exif.get("gps") if (exif.get("gps_source") in (None, "exif")) else None
    xmp = extract_xmp_gps(image)
    name = filename or _source_name(image)
    name_gps = _filename_gps(name)

    sources: list[dict[str, Any]] = []
    if exif_gps:
        sources.append(
            {
                "source": "exif",
                "label": "EXIF GPS IFD (записано камерой/дроном при съёмке)",
                "lat": exif_gps["lat"],
                "lon": exif_gps["lon"],
                "alt_m": exif_gps.get("alt_m"),
                "trusted": True,
            }
        )
    if xmp:
        sources.append(
            {
                "source": "xmp",
                "label": "XMP-пакет в JPEG (drone-dji:GPSLatitude)",
                "lat": xmp["lat"],
                "lon": xmp["lon"],
                "alt_m": xmp.get("alt_m"),
                "trusted": True,
                "keys": xmp.get("keys"),
            }
        )
    if name_gps:
        sources.append(
            {
                "source": "flight_log",
                "label": "Координаты в имени файла (выгрузка полётного лога)",
                "lat": name_gps[0],
                "lon": name_gps[1],
                "alt_m": None,
                "trusted": False,
            }
        )

    used = next((item for item in sources if item["source"] in GPS_SOURCE_PRIORITY), None)
    checks = _gps_checks(exif_gps, xmp, name_gps, _raw_exif_gps_tags(image))

    if used is None:
        explanation = (
            "Координат в файле нет: ни EXIF GPS, ни XMP, ни имени файла. "
            "Это нормально для фото из соцсетей и мессенджеров - они удаляют метаданные. "
            "Приложение не определяет геолокацию по IP, картам или устройству."
        )
    elif used["source"] == "exif":
        explanation = (
            "Координаты взяты из EXIF GPS IFD самого файла. Это метаданные, записанные "
            "при съёмке; они доказывают только то, что было записано, и НЕ доказывают, "
            "что снимок сделан по этому адресу и сегодня."
        )
    elif used["source"] == "xmp":
        explanation = (
            "EXIF GPS отсутствует, координаты взяты из XMP-пакета внутри файла. "
            "Источник тот же - метаданные съёмки, но их проще подделать, чем EXIF+QR."
        )
    else:
        explanation = (
            "Координаты взяты из имени файла (выгрузка полётного лога), а не из самого снимка. "
            "Имя файла можно переименовать, поэтому такие координаты - вспомогательные."
        )

    return {
        "used": {
            "source": used["source"] if used else "none",
            "lat": used["lat"] if used else None,
            "lon": used["lon"] if used else None,
            "alt_m": used["alt_m"] if used else None,
        },
        "sources": sources,
        "checks": checks,
        "filename": name,
        "xmp_metadata": {
            "captured_at_raw": xmp.get("captured_at_raw") if xmp else None,
            "camera_make": xmp.get("camera_make") if xmp else None,
            "drone_model": xmp.get("drone_model") if xmp else None,
        },
        "explanation": explanation,
    }


def _raw_exif_gps_tags(image: Any) -> dict[str, Any]:
    """Raw GPS IFD tags for diagnostics (GPSStatus, GPSMapDatum, ...)."""
    try:
        data = _read_source_bytes(image)
        with Image.open(io.BytesIO(data)) as opened:
            exif = opened.getexif()
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            except Exception:
                raw = exif.get(34853)
                gps_ifd = raw if isinstance(raw, dict) else {}
            return {ExifTags.GPSTAGS.get(key, str(key)): value for key, value in (gps_ifd or {}).items()}
    except Exception:
        return {}


def extract_gps_and_exif(image_file: Any, filename: str = "") -> dict[str, Any]:
    """Compatibility wrapper used by the existing Streamlit UI.

    Returns the legacy fields (`type`, `has_exif`, `lat`, `lon`, `date`,
    `drone_model`) and also the new `source` field plus the full provenance.
    """
    exif = extract_exif_data(image_file)
    if exif.get("gps"):
        gps = exif["gps"]
        return {
            "type": "Тип А (полный EXIF + GPS)",
            "has_exif": bool(exif.get("has_exif")),
            "lat": gps["lat"],
            "lon": gps["lon"],
            "date": _format_exif_date(exif.get("captured_at")),
            "drone_model": exif.get("drone_model") or "БПЛА подрядчика",
            "source": exif.get("gps_source") or "exif",
        }

    # Type B: coordinates are supplied by the flight log / file name.
    match = re.search(r"(\d{2}\.\d{3,})[_-](\d{2}\.\d{3,})", filename or "")
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return {
                "type": "Тип Б (GPS из полетного лога)",
                "has_exif": False,
                "lat": lat,
                "lon": lon,
                "date": "Из полетного задания",
                "drone_model": "БПЛА подрядчика",
                "source": "flight_log",
            }

    return {
        "type": "Метаданные отсутствуют",
        "has_exif": False,
        "lat": None,
        "lon": None,
        "date": "Неизвестно",
        "drone_model": "Неизвестно",
        "source": "none",
    }


def compute_image_hash(path: str | os.PathLike[str] | Any) -> str:
    """Return the SHA-256 hex digest of the image bytes.

    Accepts a filesystem path, bytes, or a file-like object (for Streamlit).
    The hash covers the whole file, including EXIF, so any metadata edit
    changes the digest.
    """
    return hashlib.sha256(_read_source_bytes(path)).hexdigest()


def verify_exif_gps_match(
    image_path: str | os.PathLike[str] | Any,
    cadastral_gps: tuple[float, float] | Sequence[float] | str,
    tolerance_m: float = 30.0,
) -> bool:
    """Check whether the EXIF GPS point is close to the cadastral point.

    Returns False when EXIF/GPS is absent or when the point is farther than
    `tolerance_m`. The function is deliberately conservative: a missing GPS
    never counts as a match.
    """
    if tolerance_m < 0:
        raise ValueError("tolerance_m must be non-negative")
    cadastral = _coerce_gps(cadastral_gps)
    if cadastral is None:
        return False
    exif = extract_exif_data(image_path)
    gps = exif.get("gps")
    if not gps:
        return False
    distance = _haversine_m(gps["lat"], gps["lon"], cadastral[0], cadastral[1])
    return distance <= float(tolerance_m)


# ---------------------------------------------------------------------------
# Challenge marker
# ---------------------------------------------------------------------------

_NAMED_COLORS = {
    "magenta": (255, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "cyan": (0, 255, 255),
}


def _parse_color_marker(expected_marker: str | None) -> tuple[int, int, int] | None:
    if not expected_marker:
        return None
    text = str(expected_marker).strip().lower()
    if not text.startswith("color:"):
        return None
    value = text.split(":", 1)[1].strip()
    if value in _NAMED_COLORS:
        return _NAMED_COLORS[value]
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return None
    try:
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        return r, g, b
    except ValueError:
        return None


def _has_color_marker(image_bgr: np.ndarray, rgb: tuple[int, int, int]) -> bool:
    """Detect a sufficiently large patch of the requested colour."""
    target_bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    hue = int(target_hsv[0])
    lower = np.array([max(0, hue - 10), 120, 80], dtype=np.uint8)
    upper = np.array([min(179, hue + 10), 255, 255], dtype=np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if ratio < 0.0005:
        return False
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(400.0, mask.size * 0.0005)
    return any(cv2.contourArea(contour) >= min_area for contour in contours)


def _decode_qr_codes(image_bgr: np.ndarray) -> list[str]:
    detector = cv2.QRCodeDetector()
    try:
        ok, decoded_info, _points, _straight = detector.detectAndDecodeMulti(image_bgr)
    except (cv2.error, ValueError, TypeError):
        return []
    if not ok or decoded_info is None:
        return []
    return [str(item).strip() for item in decoded_info if item]


def check_challenge_marker(image_path: str | os.PathLike[str] | Any, expected_marker: str) -> bool:
    """Check a QR challenge marker (or a colour marker stub).

    `expected_marker` is normally the random value issued for the flight task.
    For a colour demo use `"color:magenta"` or `"color:#FF00FF"`.

    The QR branch is real (OpenCV decode). The colour branch is an explicitly
    labelled demo stub and must not be treated as cryptographic proof.
    """
    image = _decode_image(image_path)
    if image is None:
        return False

    expected = (expected_marker or "").strip()
    decoded = _decode_qr_codes(image)
    if decoded:
        if not expected:
            return True
        expected_norm = expected.casefold()
        if any(value.casefold() == expected_norm for value in decoded):
            return True

    color = _parse_color_marker(expected)
    if color is not None:
        return _has_color_marker(image, color)
    return False


# ---------------------------------------------------------------------------
# Snapshot passport
# ---------------------------------------------------------------------------


def build_snapshot_passport(
    image_path: str | os.PathLike[str] | Any,
    client_id: str,
    drone_id: str,
    flight_task_id: str,
    operator_id: str = "unknown",
) -> dict[str, Any]:
    """Build the "snapshot passport" (chain-of-custody JSON).

    The passport contains no direct client PII: `client_id` must already be a
    pseudonym. The file hash is calculated over the original bytes, so any
    later edit to the image or its EXIF invalidates the passport.
    """
    raw = _read_source_bytes(image_path)
    digest = hashlib.sha256(raw).hexdigest()
    exif = extract_exif_data(image_path)
    name = _source_name(image_path)

    captured_at = exif.get("captured_at")
    captured_at_source = "exif" if captured_at else "unknown"
    if not captured_at and isinstance(image_path, (str, os.PathLike)):
        try:
            mtime = dt.datetime.fromtimestamp(Path(image_path).stat().st_mtime, tz=dt.timezone.utc)
            captured_at = mtime.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            captured_at_source = "file_mtime"
        except OSError:
            pass

    base_url = os.getenv("PASSPORT_VERIFY_BASE_URL", "https://stroykontur.local/verify").rstrip("/")
    verification_token = digest[:24]
    created_at = _utc_now_iso()

    passport: dict[str, Any] = {
        "schema_version": PASSPORT_SCHEMA_VERSION,
        "passport_id": uuid.uuid4().hex,
        "created_at": created_at,
        "client_id": str(client_id),          # pseudonym, not a full name
        "drone_id": str(drone_id),
        "flight_task_id": str(flight_task_id),
        "operator_id": str(operator_id),
        "image": {
            "filename": name,
            "size_bytes": len(raw),
            "content_type": _guess_content_type(name),
        },
        "hash": {
            "algorithm": "SHA-256",
            "value": digest,
            "scope": "file_bytes",
        },
        "captured_at": captured_at,
        "captured_at_source": captured_at_source,
        "gps": exif.get("gps"),
        "exif": {
            "drone_model": exif.get("drone_model"),
            "camera_make": exif.get("camera_make"),
            "datetime_original_raw": exif.get("captured_at_raw"),
        },
        "verification_url": f"{base_url}/{verification_token}",
        "qr_payload": f"{base_url}/{verification_token}",
        "custody": [
            {
                "step": "capture",
                "actor": str(operator_id),
                "timestamp": captured_at,
            },
            {
                "step": "passport_created",
                "actor": "system",
                "timestamp": created_at,
            },
        ],
        "signature_alg": None,
        "signature": None,
    }

    signing_key = os.getenv("PASSPORT_SIGNING_KEY")
    if signing_key:
        passport["signature_alg"] = "HMAC-SHA256"
        passport["signature"] = hmac.new(
            signing_key.encode("utf-8"),
            _canonical_json(passport),
            hashlib.sha256,
        ).hexdigest()

    return passport


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _last_record_hash(log_path: str | os.PathLike[str]) -> str:
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return AUDIT_GENESIS
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return AUDIT_GENESIS
    try:
        record = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise AuditChainError("Audit log ends with invalid JSON") from exc
    record_hash = record.get("record_hash")
    if not isinstance(record_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", record_hash):
        raise AuditChainError("Audit log ends with an invalid record_hash")
    return record_hash


def write_audit_log(event: dict[str, Any], log_path: str = "audit_trail.jsonl") -> dict[str, Any]:
    """Append one event to the append-only JSONL audit trail.

    Required event keys: actor, action, object_hash, decision.
    The function adds timestamp, hash_prev and record_hash. All other keys are
    stored under `details` so the schema stays predictable.
    """
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping/dict")

    missing = [field for field in _REQUIRED_AUDIT_FIELDS if not str(event.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Audit event is missing required fields: {', '.join(missing)}")

    object_hash = str(event["object_hash"]).strip()
    if not re.fullmatch(r"[a-f0-9]{64}", object_hash):
        raise ValueError("object_hash must be a lowercase SHA-256 hex digest (64 characters)")

    timestamp = str(event.get("timestamp") or _utc_now_iso())
    try:
        dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO 8601 date-time") from exc

    reserved = {
        "schema_version",
        "record_id",
        "timestamp",
        "actor",
        "action",
        "object_hash",
        "decision",
        "hash_prev",
        "record_hash",
        "details",
    }
    details: dict[str, Any] = {}
    nested_details = event.get("details")
    if isinstance(nested_details, Mapping):
        # The Streamlit app passes one nested `details` object. Merge it into
        # the record instead of silently dropping it.
        details.update(
            {key: value for key, value in nested_details.items() if key not in reserved}
        )
    details.update(
        {key: value for key, value in event.items() if key not in reserved}
    )

    with _AUDIT_LOCK:
        previous_hash = _last_record_hash(log_path)
        record: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "record_id": uuid.uuid4().hex,
            "timestamp": timestamp,
            "actor": str(event["actor"]),
            "action": str(event["action"]),
            "object_hash": object_hash,
            "decision": str(event["decision"]),
            "hash_prev": previous_hash,
        }
        if details:
            record["details"] = details
        record["record_hash"] = _record_hash(record)

        path = Path(log_path)
        if path.parent and str(path.parent) not in (".", ""):
            path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(path, flags, 0o600)
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return record


def verify_audit_chain(log_path: str = "audit_trail.jsonl") -> bool:
    """Return True only if every record and every link is intact."""
    path = Path(log_path)
    if not path.exists():
        return True
    expected_previous = AUDIT_GENESIS
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    return False
                if record.get("hash_prev") != expected_previous:
                    return False
                stored_hash = record.get("record_hash")
                if not isinstance(stored_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", stored_hash):
                    return False
                if stored_hash != _record_hash(record):
                    return False
                expected_previous = stored_hash
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


def audit_chain_report(log_path: str = "audit_trail.jsonl") -> dict[str, Any]:
    """Small helper for the UI: status, record count and current head."""
    path = Path(log_path)
    if not path.exists():
        return {"valid": True, "records": 0, "head": AUDIT_GENESIS, "error": None}
    expected_previous = AUDIT_GENESIS
    records = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records += 1
                if not isinstance(record, dict):
                    return {"valid": False, "records": records, "head": expected_previous, "error": "record is not an object"}
                if record.get("hash_prev") != expected_previous:
                    return {"valid": False, "records": records, "head": expected_previous, "error": "hash_prev mismatch"}
                stored_hash = record.get("record_hash")
                if not isinstance(stored_hash, str) or stored_hash != _record_hash(record):
                    return {"valid": False, "records": records, "head": expected_previous, "error": "record_hash mismatch"}
                expected_previous = stored_hash
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"valid": False, "records": records, "head": expected_previous, "error": str(exc)}
    return {"valid": True, "records": records, "head": expected_previous, "error": None}


# ---------------------------------------------------------------------------
# Compatibility quality gate for the existing UI
# ---------------------------------------------------------------------------


def run_quality_gate(
    image_file: Any,
    filename: str = "",
    expected_marker: str | None = None,
    cadastral_gps: tuple[float, float] | Sequence[float] | str | None = None,
    allow_exif_fallback: bool = True,
    tolerance_m: float = 30.0,
) -> dict[str, Any]:
    """Run sharpness, GPS and challenge-marker checks for one image.

    `allow_exif_fallback=True` preserves the old demo behaviour: a photo with
    EXIF but no visible QR is treated as having a freshness signal. For a
    stricter production mode pass False; then only a real QR/colour marker
    counts as `challenge_marker_present`.
    """
    if tolerance_m < 0:
        raise ValueError("tolerance_m must be non-negative")
    raw = _read_source_bytes(image_file)
    file_hash = hashlib.sha256(raw).hexdigest()
    image = _decode_image(raw)

    if image is None:
        return {
            "image_quality_passed": False,
            "exif_gps_match": False,
            "challenge_marker_present": False,
            "error_message": "Файл не является изображением или повреждён",
            "_meta": {
                "sha256": file_hash,
                "sha256_short": file_hash[:16],
                "sharpness": 0.0,
                "brightness": 0.0,
                "gps": extract_gps_and_exif(raw, filename),
                "gps_provenance": gps_provenance(raw, filename),
                "marker_source": "none",
                "marker_payload": None,
                "expected_marker": expected_marker,
            },
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    image_quality_passed = bool(sharpness > 120.0 and 35.0 < brightness < 230.0)

    gps_info = extract_gps_and_exif(raw, filename)
    if gps_info.get("lat") is not None and gps_info.get("lon") is not None:
        if cadastral_gps is not None:
            cadastral = _coerce_gps(cadastral_gps)
            exif_gps_match = bool(
                cadastral
                and _haversine_m(gps_info["lat"], gps_info["lon"], cadastral[0], cadastral[1])
                <= float(tolerance_m)
            )
        else:
            # Region-level fallback used by the current demo UI.
            exif_gps_match = bool(
                53.0 <= gps_info["lat"] <= 58.5 and 81.0 <= gps_info["lon"] <= 87.0
            )
    else:
        exif_gps_match = False

    decoded = _decode_qr_codes(image)
    marker_payload = decoded[0] if decoded else None
    marker_found = False
    if expected_marker:
        marker_found = check_challenge_marker(raw, expected_marker)
    else:
        marker_found = bool(decoded)

    marker_source = "qr" if marker_found else "none"
    challenge_marker_present = bool(marker_found)
    if not marker_found and allow_exif_fallback and gps_info.get("has_exif"):
        challenge_marker_present = True
        marker_source = "exif_fallback"

    errors: list[str] = []
    if not image_quality_passed:
        errors.append(f"Низкое качество кадра: резкость={sharpness:.0f}, яркость={brightness:.0f}")
    if not exif_gps_match:
        errors.append("GPS отсутствует или не совпадает с кадастровым округом")
    if not challenge_marker_present:
        errors.append("Не подтверждена свежесть снимка: нет QR-маркера и EXIF")

    return {
        "image_quality_passed": image_quality_passed,
        "exif_gps_match": exif_gps_match,
        "challenge_marker_present": challenge_marker_present,
        "error_message": "; ".join(errors) if errors else None,
        "_meta": {
            "sha256": file_hash,
            "sha256_short": file_hash[:16],
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "gps": gps_info,
            "gps_provenance": gps_provenance(raw, filename),
            "marker_source": marker_source,
            "marker_payload": marker_payload,
            "expected_marker": expected_marker,
        },
    }


# ---------------------------------------------------------------------------
# Evidence level: насколько доказан сам снимок
# ---------------------------------------------------------------------------
#
# Бизнес-заказчик подтвердил на Q&A важную вещь: **дрон не подписывает снимки**
# цифровой подписью, а на части снимков метаданных нет вообще. Значит нельзя
# строить решение на одном EXIF: он и не подписан, и может отсутствовать.
#
# Поэтому снимок оценивается не «хороший/плохой», а по уровню доказанности:
#
#   strong  - есть GPS и настоящий QR-маркер полётного задания;
#   medium  - есть GPS, свежесть подтверждена только EXIF (демо-режим);
#   weak    - есть что-то одно: GPS без маркера, маркер без GPS, только XMP
#             или имя файла;
#   none    - нет ни GPS, ни маркера.
#
# Дирижёр не называет «weak» фродом: он отправляет снимок человеку. Фрод
# ставится только при «none», когда нет вообще ничего.

EVIDENCE_LEVELS = ("strong", "medium", "weak", "none")


def evidence_level(
    quality_gate: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Оценить, насколько доказан снимок (не подпись дрона, а обвязка вокруг)."""
    meta = meta or {}
    gps_ok = bool(quality_gate.get("exif_gps_match"))
    marker_present = bool(quality_gate.get("challenge_marker_present"))
    marker_source = str(meta.get("marker_source") or ("qr" if marker_present else "none"))

    if marker_source == "exif_fallback":
        marker_kind = "exif_fallback"
    elif marker_present:
        marker_kind = "qr"
    else:
        marker_kind = "none"

    if gps_ok and marker_kind == "qr":
        level, reason = "strong", "GPS из метаданных + QR-маркер полётного задания"
    elif gps_ok and marker_kind == "exif_fallback":
        level, reason = "medium", "GPS есть, свежесть подтверждена только EXIF (демо-режим)"
    elif gps_ok:
        level, reason = "weak", "GPS есть, но свежесть снимка ничем не подтверждена"
    elif marker_kind == "qr":
        level, reason = "weak", "QR-маркер есть, но геопривязка отсутствует"
    elif marker_kind == "exif_fallback":
        level, reason = "weak", "EXIF без GPS и без QR-маркера"
    else:
        level, reason = "none", "нет ни GPS, ни QR-маркера"

    return {
        "level": level,
        "reason": reason,
        "gps_confirmed": gps_ok,
        "marker": marker_kind,
        "note": (
            "Дрон не подписывает снимок. Уровень описывает обвязку: "
            "метаданные + маркер полётного задания + журнал, а не подпись камеры."
        ),
    }


__all__ = [
    "AUDIT_GENESIS",
    "AuditChainError",
    "EVIDENCE_LEVELS",
    "audit_chain_report",
    "build_snapshot_passport",
    "check_challenge_marker",
    "compute_image_hash",
    "evidence_level",
    "extract_exif_data",
    "extract_gps_and_exif",
    "extract_xmp_gps",
    "gps_provenance",
    "run_quality_gate",
    "verify_audit_chain",
    "verify_exif_gps_match",
    "write_audit_log",
]

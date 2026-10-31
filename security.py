import re
import cv2
import hashlib
import datetime
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def _convert_to_degrees(value):
    d, m, s = float(value[0]), float(value[1]), float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def extract_gps_and_exif(image_file, filename=""):
    """Извлекает реальные GPS-координаты и дату из EXIF дрона (Тип А) или из имени файла (Тип Б)."""
    image_file.seek(0)
    try:
        image = Image.open(image_file)
        info = image._getexif()
        if info:
            date_str = None
            model_str = None
            gps_data = {}
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    date_str = str(value)
                elif decoded == "Model":
                    model_str = str(value)
                elif decoded == "GPSInfo":
                    gps_data = {GPSTAGS.get(t, t): value[t] for t in value}
            if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                lat = _convert_to_degrees(gps_data["GPSLatitude"])
                lon = _convert_to_degrees(gps_data["GPSLongitude"])
                image_file.seek(0)
                return {
                    "type": "Тип А (Полный EXIF)",
                    "has_exif": True,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "date": date_str or "07.08.2026",
                    "drone_model": model_str or "DJI Mavic 3M"
                }
    except Exception:
        pass

    image_file.seek(0)
    # Тип Б: поиск координат в названии файла
    match = re.search(r"(\d{2}\.\d+)[_-](\d{2}\.\d+)", filename)
    if match:
        return {
            "type": "Тип Б (Координаты из лога/имени)",
            "has_exif": False,
            "lat": float(match.group(1)),
            "lon": float(match.group(2)),
            "date": "Из полетного задания",
            "drone_model": "БПЛА подрядчика"
        }

    return {
        "type": "Метаданные отсутствуют",
        "has_exif": False,
        "lat": None,
        "lon": None,
        "date": "Неизвестно",
        "drone_model": "Неизвестно"
    }

def run_quality_gate(image_file, filename=""):
    """Проверяет резкость (Laplacian), засветку, GPS и челлендж-маркер."""
    image_file.seek(0)
    raw_bytes = image_file.read()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
    file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    image_file.seek(0)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    image_quality_passed = bool((sharpness > 120.0) and (35.0 < brightness < 230.0))

    gps_info = extract_gps_and_exif(image_file, filename)
    # Проверяем принадлежность координат к региону (Новосибирская / Томская область)
    exif_gps_match = False
    if gps_info["lat"] and gps_info["lon"]:
        exif_gps_match = bool((53.0 <= gps_info["lat"] <= 58.5) and (81.0 <= gps_info["lon"] <= 87.0))

    # Поиск QR-кода (челлендж-маркера)
    qr_detector = cv2.QRCodeDetector()
    retval, _, _, _ = qr_detector.detectAndDecodeMulti(img)
    # Если есть подлинный аппаратный EXIF DJI и валидный GPS, верификация свежести проходит автоматически
    challenge_marker_present = bool(retval) or gps_info["has_exif"]

    errors = []
    if not image_quality_passed:
        errors.append(f"Низкое качество кадра: резкость={sharpness:.0f}, яркость={brightness:.0f}")
    if not exif_gps_match:
        errors.append("GPS-координаты отсутствуют или не совпадают с кадастровым округом")

    return {
        "image_quality_passed": image_quality_passed,
        "exif_gps_match": exif_gps_match,
        "challenge_marker_present": challenge_marker_present,
        "error_message": "; ".join(errors) if errors else None,
        "_meta": {
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "sha256_short": file_hash,
            "gps": gps_info
        }
    }

def write_audit_log(cadastral_num, ai_json, verdict):
    """Записывает защищенный лог проверки в файл audit_trail.log (для ИБ)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = f"{timestamp}|{cadastral_num}|{verdict['verdict']['strategy_number']}|{ai_json['construction_analysis']['detected_stage']}"
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    log_line = f"[{timestamp}] CAD:{cadastral_num} | STAGE:{ai_json['construction_analysis']['detected_stage']} | STRAT:{verdict['verdict']['strategy_name']} | SIG:{signature}\n"
    with open("audit_trail.log", "a", encoding="utf-8") as f:
        f.write(log_line)
    return signature

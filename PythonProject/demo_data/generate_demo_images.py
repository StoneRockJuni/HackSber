"""Generate three demo aerials for the hackathon.

The images are synthetic: no real client data, no real coordinates of a
private person. They exist only to demonstrate the security pipeline on a
laptop without a drone.

Run from the project root:

    .\\.venv\\Scripts\\python.exe demo_data\\generate_demo_images.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MARKER = "STROYKONTUR-DEMO-2026"
ROOT = Path(__file__).resolve().parent


def _qr() -> np.ndarray:
    encoder = cv2.QRCodeEncoder_create()
    qr = encoder.encode(MARKER)
    return cv2.resize(qr, (300, 300), interpolation=cv2.INTER_NEAREST)


def _base_scene(title: str, house: bool) -> np.ndarray:
    image = np.full((1200, 1600, 3), (95, 140, 80), dtype=np.uint8)
    # Ground / driveway
    cv2.rectangle(image, (0, 900), (1600, 1200), (120, 115, 105), -1)
    if house:
        cv2.rectangle(image, (300, 300), (1000, 850), (215, 215, 215), -1)
        cv2.rectangle(image, (320, 320), (980, 830), (150, 150, 150), -1)
        pts = np.array([[250, 330], [650, 120], [1050, 330]], dtype=np.int32)
        cv2.fillPoly(image, [pts], (80, 80, 80))
        cv2.rectangle(image, (560, 650), (720, 830), (80, 90, 110), -1)
        cv2.rectangle(image, (380, 480), (530, 620), (110, 140, 170), -1)
    else:
        cv2.rectangle(image, (300, 500), (850, 900), (180, 180, 180), -1)
        cv2.rectangle(image, (330, 530), (820, 870), (135, 135, 135), -1)
        for x in range(360, 800, 90):
            cv2.rectangle(image, (x, 560), (x + 55, 840), (190, 160, 70), -1)
    cv2.putText(image, title, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    qr = cv2.cvtColor(_qr(), cv2.COLOR_GRAY2BGR)
    image[850:1150, 1220:1520] = qr
    return image


def _save_with_exif(image_bgr: np.ndarray, path: Path, lat: float, lon: float, captured_at: str) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    exif = Image.Exif()

    def dms(value: float) -> tuple[int, int, float]:
        absolute = abs(value)
        degrees = int(absolute)
        minutes = int((absolute - degrees) * 60)
        seconds = round(((absolute - degrees) * 60 - minutes) * 60, 2)
        return degrees, minutes, seconds

    exif[271] = "DJI"
    exif[272] = "Mavic 3M"
    exif[36867] = captured_at
    exif[34853] = {
        1: "N" if lat >= 0 else "S",
        2: dms(lat),
        3: "E" if lon >= 0 else "W",
        4: dms(lon),
        5: 0,
        6: 142.3,
    }
    pil.save(path, exif=exif, quality=92)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _save_with_exif(
        _base_scene("DEMO 1 - foundation / 50%", house=False),
        ROOT / "demo_object_1_ok.jpg",
        54.8511,
        83.1946,
        "2026:10:31 09:42:17",
    )
    _save_with_exif(
        _base_scene("DEMO 2 - house / 100%", house=True),
        ROOT / "demo_object_2_ok.jpg",
        54.8370,
        83.2301,
        "2026:10:31 10:05:44",
    )
    # Fraud demo: no EXIF, no QR. The gate must route it to strategy 6.
    # Add high-frequency texture so the image passes the sharpness check; the
    # point of the demo is the missing GPS/marker, not a blurry frame.
    rng = np.random.default_rng(42)
    fraud = rng.integers(0, 255, size=(1200, 1600, 3), dtype=np.uint8)
    cv2.rectangle(fraud, (300, 300), (1000, 900), (130, 130, 130), -1)
    cv2.putText(fraud, "DEMO 3 - no EXIF / no QR", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(ROOT / "demo_fraud_no_exif.jpg"), fraud)
    print("Demo images written to", ROOT)


if __name__ == "__main__":
    main()

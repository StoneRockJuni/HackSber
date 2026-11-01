"""SberStroi YOLOv8 adapter for the construction-stage agent.

The original `SberStroi.ipynb` trained a YOLOv8n detector on six classes:
doors, fence, foundation, roof_finished, walls, windows. It then mapped the
detected classes to a 0/50/100% construction stage.

This module keeps that useful part but adds three things needed for the bank
demo:

1. **Optional runtime** — if `models/sberstroi_yolov8n.onnx` exists, it is used;
   otherwise the caller falls back to the old OpenCV logic and the app still
   starts.
2. **No leakage in the demo split** — training uses whole flight sequences for
   validation (`SberStroi_dataset/prepare_yolo_split.py`).
3. **Uncertainty is explicit** — no detections means "manual review", not a
   confident 0% house.

The ONNX file is decoded with `cv2.dnn`, so the Streamlit app does not need
PyTorch or Ultralytics at runtime.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "sberstroi_yolov8n.onnx"
CLASS_NAMES = ["doors", "fence", "foundation", "roof_finished", "walls", "windows"]
CLASS_NAMES_RU = {
    "doors": "двери",
    "fence": "забор",
    "foundation": "фундамент",
    "roof_finished": "готовая кровля",
    "walls": "стены",
    "windows": "окна",
}
INPUT_SIZE = 640
MANUAL_REVIEW_THRESHOLD = 0.75

# The ONNX graph is ~13 MB; parsing it for every frame of a 40-shot flight is
# pure waste. Cache one network per (path, mtime, size) and serialise forward
# passes, because cv2.dnn.Net keeps the last input in its own state.
_NET_CACHE: dict[tuple[str, float, int], Any] = {}
_NET_LOCK = threading.Lock()


def _load_net(path: Path) -> Any:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _NET_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        net = cv2.dnn.readNetFromONNX(str(path))
    except cv2.error:
        # Some OpenCV Windows builds do not accept non-ASCII paths.
        net = cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(), dtype=np.uint8))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    _NET_CACHE.clear()
    _NET_CACHE[key] = net
    return net


def model_path() -> Path:
    configured = os.getenv("SBER_YOLO_MODEL", "").strip()
    return Path(configured) if configured else DEFAULT_MODEL


def model_available() -> bool:
    return model_path().is_file()


def model_sha256() -> str | None:
    path = model_path()
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_bytes(source: Any) -> bytes:
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        try:
            source.seek(0)
        except Exception:
            pass
        data = source.read()
        try:
            source.seek(0)
        except Exception:
            pass
        return bytes(data)
    raise TypeError("source must be a path, bytes, or a binary file-like object")


def _letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx = (size - new_w) // 2
    dy = (size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas, scale, dx, dy


def _decode_output(
    output: np.ndarray,
    scale: float,
    dx: int,
    dy: int,
    original_width: int,
    original_height: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    prediction = np.asarray(output)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        return []

    expected = 4 + len(CLASS_NAMES)
    if prediction.shape[0] == expected and prediction.shape[1] != expected:
        prediction = prediction.T
    if prediction.shape[1] < expected:
        return []

    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for row in prediction:
        class_scores = row[4 : 4 + len(CLASS_NAMES)]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < conf_threshold:
            continue
        cx, cy, width, height = (float(value) for value in row[:4])
        x = (cx - width / 2 - dx) / scale
        y = (cy - height / 2 - dy) / scale
        w = width / scale
        h = height / scale
        if w <= 1 or h <= 1:
            continue
        boxes.append([x, y, w, h])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    result: list[dict[str, Any]] = []
    for index in np.asarray(indices).flatten().tolist():
        x, y, w, h = boxes[index]
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(original_width, int(round(x + w)))
        y2 = min(original_height, int(round(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue
        class_name = CLASS_NAMES[class_ids[index]]
        result.append(
            {
                "class_name": class_name,
                "class_name_ru": CLASS_NAMES_RU[class_name],
                "confidence": round(float(scores[index]), 4),
                "box": [x1, y1, x2, y2],
            }
        )
    return result


def detect_elements(
    source: Any,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]] | None:
    """Run the ONNX detector. Returns None when the model is not installed."""
    if not model_path().is_file():
        return None

    raw = _read_bytes(source)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    return detect_elements_array(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)


def detect_elements_array(
    image: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]] | None:
    """Run the ONNX detector on an already decoded BGR frame.

    Kept separate from :func:`detect_elements` so callers that work with a
    series of frames (or need to crop/zoom one frame) decode the JPEG only
    once. Returns None when the model is not installed.
    """
    path = model_path()
    if not path.is_file():
        return None
    if image is None or getattr(image, "size", 0) == 0:
        return []

    canvas, scale, dx, dy = _letterbox(image, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(
        canvas,
        scalefactor=1.0 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    try:
        net = _load_net(path)
        # cv2.dnn.Net is stateful (setInput/forward), so a series can reuse one
        # instance only under a lock.
        with _NET_LOCK:
            net.setInput(blob)
            output = net.forward()
    except Exception:
        # A corrupt/substituted ONNX must not crash the upload pipeline.
        # Return no detections; the caller will require manual review.
        return []
    height, width = image.shape[:2]
    return _decode_output(
        output,
        scale,
        dx,
        dy,
        width,
        height,
        conf_threshold,
        iou_threshold,
    )


def stage_from_classes(classes: set[str], confidence_by_class: dict[str, float] | None = None) -> tuple[str, float]:
    """Map SberStroi classes to a conservative 0/50/100% stage.

    The original notebook used different rules in different cells. This version
    only auto-confirms 100% when at least two independent finishing signals are
    visible (for example, a roof plus walls). Roof-only evidence is ambiguous:
    it can be a finished house, but it can also be a neighbouring roof at the
    edge of an aerial frame. Roof-only therefore returns 50% with low
    confidence, which the Director routes to manual review.
    """
    confidence_by_class = confidence_by_class or {}
    finishing = {"roof_finished", "windows", "doors"}
    structural = {"foundation", "walls"}

    has_roof = "roof_finished" in classes
    has_windows = "windows" in classes
    has_doors = "doors" in classes
    has_walls = "walls" in classes
    has_foundation = "foundation" in classes
    strongest = max(confidence_by_class.values(), default=0.0)

    # Closed house: roof plus a second independent signal.
    if has_roof and (has_walls or has_windows or has_doors):
        base = 0.72 + 0.04 * len(classes & finishing) + 0.04 * len(classes & structural)
        return "100%", min(0.98, base + 0.1 * strongest)

    # Windows plus doors is also a strong closed-contour signal.
    if has_windows and has_doors:
        base = 0.70 + 0.04 * len(classes & finishing)
        return "100%", min(0.96, base + 0.1 * strongest)

    # Roof-only / roof+foundation without walls/windows/doors: do not claim
    # 100% automatically. This is deliberately below the Director's manual
    # review threshold (0.75).
    if has_roof:
        return "50%", 0.45

    # Foundation or walls alone: plausible 50%, but still a low-confidence
    # automatic claim. The Director will ask for a human/extra flight.
    if has_walls or has_foundation:
        base = 0.60 + 0.03 * len(classes & structural)
        return "50%", min(0.74, base + 0.08 * strongest)

    # Fence-only / no relevant elements. This is not proof that construction
    # has not started, so confidence is deliberately low.
    return "0%", 0.35 if "fence" in classes else 0.0


def analyze_construction(source: Any, conf_threshold: float = 0.25) -> dict[str, Any] | None:
    """Return a construction analysis block, or None if the model is absent."""
    detections = detect_elements(source, conf_threshold=conf_threshold)
    if detections is None:
        return None

    classes = {item["class_name"] for item in detections}
    confidence_by_class = {item["class_name"]: item["confidence"] for item in detections}
    stage, stage_confidence = stage_from_classes(classes, confidence_by_class)
    ordered = [name for name in CLASS_NAMES if name in classes]
    human = ", ".join(CLASS_NAMES_RU[name] for name in ordered) if ordered else "ничего релевантного"
    needs_review = not detections or stage_confidence < MANUAL_REVIEW_THRESHOLD

    if needs_review:
        notes = (
            "YOLOv8 не нашёл устойчивых элементов дома (фундамент/стены/кровля/окна/двери). "
            "Автоматическая стадия не подтверждена, нужен ручной разбор или повторный облёт."
        )
    else:
        notes = (
            f"YOLOv8 SberStroi обнаружил: {human}. "
            f"Расчётная стадия — {stage} (уверенность {stage_confidence:.2f}). "
            "Материал стен по детектору не определяется; границы/ВРИ требуют отдельной проверки."
        )

    return {
        "construction_analysis": {
            "detected_stage": stage,
            "confidence_stage": round(stage_confidence, 3),
            "wall_material": "unknown",
            "confidence_material": 0.0,
            "detected_classes": ordered,
            "analysis_source": "sber_yolo_onnx",
            "needs_manual_review": needs_review,
        },
        "geospatial_check": {
            "boundaries_violation": False,
            "vri_compliant": True,
            "site_development_notes": notes,
        },
        "ai_meta": {
            "source": "sber_yolo_onnx",
            "model": model_path().name,
            "model_sha256": model_sha256(),
            "conf_threshold": conf_threshold,
            "detections": detections,
            "detected_classes": ordered,
            "stage_confidence": round(stage_confidence, 3),
            "needs_manual_review": needs_review,
        },
    }

"""Object-centric construction analysis for a *series* of drone photos.

Why this module exists
----------------------
`SberStroi.ipynb` trained a six-class detector and then applied a one-line rule
per single frame: "roof/window/door seen -> 100%". The streamlit prototype took
the opposite shortcut: "no roof+wall pair -> ask a human". Both are wrong for
aerial nadir photography, and the measured behaviour on the 56 real flight
photos shows exactly why:

* a **finished** house photographed from directly above mostly shows **roof
  only** - windows and doors are simply not visible from that angle, so the
  "two independent signals" rule reports 50% for a 100% house;
* a **construction site** shows a foundation and material stacks, but the
  detector sometimes fires `roof_finished` on a **neighbouring** roof at the
  frame edge, so the notebook rule reports 100% for a 50% object;
* one frame of the same object can say 0%, the next 50%, the next 100%.

The fix is not a bigger threshold, it is a different question. Instead of
"which classes are somewhere in this JPEG", this module asks:

1. **Where is the object of interest?** The bank's drone photographs the
   pledged plot, so the object is near the frame centre. Detections at the
   edge are usually neighbours and are down-weighted, not trusted.
2. **What does the *series* say?** A flight is 5-40 frames of one plot.
   Evidence is pooled across frames, and "nothing detected" is treated as
   *absence of evidence*, never as positive proof of 0%.
3. **What is the confidence of the claim?** Stage confidence is computed from
   detector score, object size and cross-frame agreement, so a weak detection
   lands below the Director's 0.75 manual-review threshold.

Everything here is deterministic and local: no network, no cloud model.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from sber_vision import (
    CLASS_NAMES,
    MANUAL_REVIEW_THRESHOLD,
    detect_elements_array,
    model_available,
)

# --- geometry of "the object of interest" ---------------------------------
# Normalised offsets from the frame centre. A detection whose centre is
# outside this window is treated as a neighbour / background object.
CENTRAL_X = 0.30
CENTRAL_Y = 0.38
# A roof smaller than ~1.2% of the frame is too small to be the pledged object
# in a normal survey shot (it is usually a shed or a neighbour).
ROOF_MIN_AREA = 0.012
MIN_AREA = 0.0015

# Detection classes that describe the building itself.
OBJECT_CLASSES = ("roof_finished", "walls", "foundation", "windows", "doors")
# Classes that prove *something* is built but not that the contour is closed.
WEAK_CLASSES = ("windows", "doors", "fence")

# --- classical colour cues (used only as weak, always-labelled evidence) ---
GREEN_LOW = (30, 40, 40)
GREEN_HIGH = (90, 255, 255)
YELLOW_LOW = (18, 120, 120)      # loose: any yellowish surface (diagnostics)
YELLOW_HIGH = (32, 255, 255)
PALLET_LOW = (20, 160, 140)      # tight: factory-wrapped gas-concrete blocks
PALLET_HIGH = (34, 255, 255)
PALLET_BLOB = 0.004
EMPTY_GREEN_RATIO = 0.55

STAGE_100 = "100%"
STAGE_50 = "50%"
STAGE_0 = "0%"


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def decode_image(source: Any) -> np.ndarray | None:
    """Decode bytes / path / file-like into a BGR frame (None if not an image).

    `cv2.imread` cannot open Windows paths containing Cyrillic characters, so
    every read goes through `imdecode` of the raw bytes.
    """
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, os.PathLike)):
        try:
            raw = Path(source).read_bytes()
        except OSError:
            return None
    elif hasattr(source, "read"):
        try:
            source.seek(0)
        except Exception:
            pass
        raw = source.read()
        try:
            source.seek(0)
        except Exception:
            pass
    elif isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
    else:
        raise TypeError(f"unsupported image source: {type(source)!r}")
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def classical_cues(image: np.ndarray) -> dict[str, float]:
    """Cheap colour statistics. They are heuristics, not a trained model.

    ``pallet_blob`` is the largest *compact* patch of saturated yellow in the
    middle of the frame, not the total yellow area. Dry grass and sun glare
    also raise the total yellow ratio (measured: 2.3% on a finished house with
    tan surroundings), but they are scattered, so their largest blob stays
    around 0.1% - while stacked gas-concrete pallets form one 0.8-2.7% blob.
    This is what keeps the "pallets on site" cue from firing on random photos.
    """
    if image is None or image.size == 0:
        return {"green_ratio": 0.0, "yellow_ratio": 0.0, "pallet_blob": 0.0, "brightness": 0.0}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    total = float(height * width)
    green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    pallet_mask = cv2.inRange(hsv, PALLET_LOW, PALLET_HIGH)
    saturation = hsv[:, :, 1]

    cy0, cy1 = int(height * 0.25), int(height * 0.85)
    cx0, cx1 = int(width * 0.20), int(width * 0.80)
    centre = pallet_mask[cy0:cy1, cx0:cx1]
    blob = 0.0
    if centre.size:
        contours, _ = cv2.findContours(centre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob = max((cv2.contourArea(c) for c in contours), default=0.0) / float(centre.size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "green_ratio": round(float(np.count_nonzero(green)) / total, 4),
        "yellow_ratio": round(float(np.count_nonzero(yellow)) / total, 4),
        "pallet_blob": round(float(blob), 4),
        "brightness": round(float(np.mean(gray)), 1),
        # Признак «ненастоящего» цвета: снимок мультиспектральный или это
        # вегетационный индекс (NDVI-раскраска), а не обычное RGB-фото.
        "sat_median": round(float(np.median(saturation)), 1),
        "high_sat_ratio": round(float(np.mean(saturation > 200)), 4),
    }


def _box_geometry(detection: dict[str, Any], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = detection["box"]
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    return {
        "cx": (x1 + x2) / 2.0 / max(width, 1),
        "cy": (y1 + y2) / 2.0 / max(height, 1),
        "area": (box_w * box_h) / float(max(width * height, 1)),
        "width_px": box_w,
        "height_px": box_h,
    }


def _is_central(geometry: dict[str, float]) -> bool:
    return abs(geometry["cx"] - 0.5) <= CENTRAL_X and abs(geometry["cy"] - 0.5) <= CENTRAL_Y


def _centrality_weight(geometry: dict[str, float]) -> float:
    """1.0 in the very centre, falling to ~0.55 at the edge of the window."""
    distance = math.hypot(
        (geometry["cx"] - 0.5) / max(CENTRAL_X, 1e-6),
        (geometry["cy"] - 0.5) / max(CENTRAL_Y, 1e-6),
    )
    return float(max(0.55, 1.0 - 0.45 * min(distance, 1.0)))


def _position_factor(geometry: dict[str, float]) -> float:
    """Мягкий штраф за смещение внутри центральной зоны.

    Раньше множитель опускался до 0.55 и «съедал» уверенность даже у крупной
    кровли в центре кадра. Теперь центральность решается окном ``_is_central``,
    а здесь остаётся только небольшая поправка 0.90..1.00.
    """
    return 0.90 + 0.10 * _centrality_weight(geometry)


def _stage_100_confidence(score: float, area: float) -> float:
    """Roof seen on the object of interest -> closed contour.

    From a nadir view the roof is the only *reliable* finishing signal: a
    finished roof means the building volume is closed even when windows and
    doors are hidden by the angle.

    Calibration note: an earlier version was ``0.58 + 0.30*score + 1.0*area``
    and multiplied by a harsh position penalty. On real 5280x3956 frames the
    roof score of a clearly finished house is often 0.4-0.6 (small object after
    downscale), so that formula produced 0.57-0.59 - under the Director's 0.75
    threshold - and every real photo went to "repeat the flight". The score is
    the primary signal; apparent size only adds a bounded bonus.
    """
    confidence = 0.55 + 0.40 * score + 0.10 * min(area / 0.06, 1.0)
    return round(min(0.97, confidence), 3)


def _stage_50_confidence(score: float, base: float) -> float:
    return round(min(0.74, base + 0.25 * score), 3)


# ---------------------------------------------------------------------------
# Single frame
# ---------------------------------------------------------------------------


def analyze_frame(
    source: Any,
    conf_threshold: float = 0.25,
    *,
    use_zoom: bool = True,
) -> dict[str, Any] | None:
    """Analyse one frame and return stage evidence for the object of interest.

    Returns ``None`` when the ONNX model is not installed (the caller then
    keeps the legacy OpenCV fallback).

    ``use_zoom`` adds a second pass over the central 60% crop. Windows and
    doors are tiny in a 5280x3956 frame squeezed into 640 px; the zoom pass
    recovers some of them. The zoom pass may only *support* a 100% claim with
    a non-roof element - it can never invent a roof by itself.
    """
    image = decode_image(source)
    if image is None:
        return {
            "stage": STAGE_0,
            "confidence": 0.0,
            "detections": [],
            "classes": [],
            "cues": {
                "green_ratio": 0.0,
                "yellow_ratio": 0.0,
                "pallet_blob": 0.0,
                "brightness": 0.0,
                "sat_median": 0.0,
                "high_sat_ratio": 0.0,
            },
            "reason": "frame_not_decoded",
            "needs_manual_review": True,
        }

    if not model_available():
        return None

    height, width = image.shape[:2]
    detections = detect_elements_array(image, conf_threshold=conf_threshold) or []
    cues = classical_cues(image)

    # Мультиспектральные снимки и NDVI-раскраски (_F.JPG, _MS_*.TIF) — это не
    # обычное RGB-фото: цвета искусственные, поэтому и «поддоны», и «крыша» на
    # них означают ничего. Такие кадры честно помечаем и не угадываем стадию.
    if cues["high_sat_ratio"] >= 0.5 and cues["sat_median"] >= 200:
        return {
            "stage": STAGE_0,
            "confidence": 0.0,
            "detections": detections,
            "classes": [],
            "cues": cues,
            "object_box": None,
            "reason": "false_color_image",
            "needs_manual_review": True,
        }

    enriched: list[dict[str, Any]] = []
    for detection in detections:
        geometry = _box_geometry(detection, width, height)
        enriched.append(
            {
                **detection,
                "cx": round(geometry["cx"], 4),
                "cy": round(geometry["cy"], 4),
                "area": round(geometry["area"], 5),
                "central": _is_central(geometry) and geometry["area"] >= MIN_AREA,
            }
        )

    zoom_evidence: list[dict[str, Any]] = []
    if use_zoom:
        zoom_evidence = _zoom_pass(image, conf_threshold)

    return _decide_frame(enriched, zoom_evidence, cues)


def _zoom_pass(image: np.ndarray, conf_threshold: float) -> list[dict[str, Any]]:
    """Second detector pass over the central crop.

    Only *auxiliary* classes are accepted from this pass (windows, doors,
    walls, foundation). A roof found only in the zoom crop is ignored on
    purpose: crops of neighbouring buildings at the centre of a wide frame
    are exactly how false 100% claims were produced before.
    """
    height, width = image.shape[:2]
    margin_x, margin_y = int(width * 0.20), int(height * 0.20)
    crop = image[margin_y : height - margin_y, margin_x : width - margin_x]
    if crop.size == 0:
        return []
    detections = detect_elements_array(crop, conf_threshold=conf_threshold) or []
    accepted: list[dict[str, Any]] = []
    for detection in detections:
        if detection["class_name"] not in {"windows", "doors", "walls", "foundation"}:
            continue
        accepted.append({**detection, "source": "zoom"})
    return accepted


def _decide_frame(
    detections: list[dict[str, Any]],
    zoom_evidence: list[dict[str, Any]],
    cues: dict[str, float],
) -> dict[str, Any]:
    def weighted(best: dict[str, Any]) -> float:
        geometry = {
            "cx": best["cx"],
            "cy": best["cy"],
            "area": best["area"],
        }
        return _position_factor(geometry)

    central = [d for d in detections if d.get("central")]

    roofs = [d for d in central if d["class_name"] == "roof_finished" and d["area"] >= ROOF_MIN_AREA]
    if roofs:
        # Выбираем не самый большой бокс, а самый уверенный: крупная рамка с
        # score 0.27 (часто «склеенный» соседский участок) не должна затирать
        # аккуратную кровлю со score 0.64.
        scored = [
            (
                round(_stage_100_confidence(float(d["confidence"]), float(d["area"])) * _position_factor(d), 3),
                d,
            )
            for d in roofs
        ]
        confidence, best = max(scored, key=lambda item: item[0])
        classes = sorted({d["class_name"] for d in central})
        return {
            "stage": STAGE_100,
            "confidence": confidence,
            "detections": detections,
            "classes": classes,
            "cues": cues,
            "object_box": best["box"],
            "reason": "central_finished_roof",
            "needs_manual_review": confidence < MANUAL_REVIEW_THRESHOLD,
        }

    # The drone does not always keep the plot in the middle of the frame.
    # A *large* finished roof seen anywhere is still strong evidence - but it
    # is discounted by position and never auto-approved on its own, because
    # this is also how a neighbour's roof used to leak into 100%.
    big_roofs = [
        d
        for d in detections
        if d["class_name"] == "roof_finished" and d["area"] >= ROOF_MIN_AREA * 2.5 and d["confidence"] >= 0.5
    ]
    if big_roofs:
        best = max(big_roofs, key=lambda d: (d["area"], d["confidence"]))
        confidence = round(
            min(0.74, _stage_100_confidence(float(best["confidence"]), float(best["area"])) - 0.10), 3
        )
        return {
            "stage": STAGE_100,
            "confidence": confidence,
            "detections": detections,
            "classes": sorted({d["class_name"] for d in central} or {d["class_name"] for d in detections}),
            "cues": cues,
            "object_box": best["box"],
            "reason": "off_centre_large_roof",
            "needs_manual_review": True,
        }

    walls = [d for d in detections if d["class_name"] == "walls"]
    foundations = [d for d in detections if d["class_name"] == "foundation"]
    weak = [d for d in central if d["class_name"] in ("windows", "doors")]
    zoom_weak = [d for d in zoom_evidence if d["class_name"] in ("windows", "doors")]

    if walls:
        best = max(walls, key=lambda d: (d["confidence"], d["area"]))
        confidence = round(_stage_50_confidence(float(best["confidence"]), 0.52) * weighted(best), 3)
        return {
            "stage": STAGE_50,
            "confidence": confidence,
            "detections": detections,
            "classes": sorted({d["class_name"] for d in detections}),
            "cues": cues,
            "object_box": best["box"],
            "reason": "walls_no_roof",
            "needs_manual_review": confidence < MANUAL_REVIEW_THRESHOLD,
        }

    if foundations:
        best = max(foundations, key=lambda d: (d["confidence"], d["area"]))
        confidence = round(_stage_50_confidence(float(best["confidence"]), 0.50) * weighted(best), 3)
        return {
            "stage": STAGE_50,
            "confidence": confidence,
            "detections": detections,
            "classes": sorted({d["class_name"] for d in detections}),
            "cues": cues,
            "object_box": best["box"],
            "reason": "foundation",
            "needs_manual_review": confidence < MANUAL_REVIEW_THRESHOLD,
        }

    # Windows/doors in the middle of the frame: a building certainly exists,
    # but the detector did not localise a roof or walls, so the contour state
    # is unknown. This never produces an automatic 100%.
    if weak or zoom_weak:
        best = max(weak or zoom_weak, key=lambda d: d["confidence"])
        classes = sorted({d["class_name"] for d in central})
        return {
            "stage": STAGE_50,
            "confidence": 0.45,
            "detections": detections,
            "classes": classes,
            "cues": cues,
            "object_box": best["box"],
            "reason": "central_openings_only",
            "needs_manual_review": True,
        }

    # Gas-concrete pallets on site: building has started, contour is open.
    if float(cues.get("pallet_blob", 0.0)) >= PALLET_BLOB:
        return {
            "stage": STAGE_50,
            "confidence": 0.58,
            "detections": detections,
            "classes": sorted({d["class_name"] for d in central}),
            "cues": cues,
            "object_box": None,
            "reason": "pallets_on_site",
            "needs_manual_review": True,
        }

    # Nothing relevant in the centre of the frame.
    empty_like = float(cues.get("green_ratio", 0.0)) >= EMPTY_GREEN_RATIO
    fence_seen = any(d["class_name"] == "fence" for d in detections)
    if empty_like or fence_seen:
        return {
            "stage": STAGE_0,
            "confidence": 0.55 if empty_like else 0.40,
            "detections": detections,
            "classes": sorted({d["class_name"] for d in detections}),
            "cues": cues,
            "object_box": None,
            "reason": "no_object_vegetated" if empty_like else "fence_only",
            "needs_manual_review": True,
        }

    return {
        "stage": STAGE_0,
        "confidence": 0.0,
        "detections": detections,
        "classes": [],
        "cues": cues,
        "object_box": None,
        "reason": "nothing_detected",
        "needs_manual_review": True,
    }


# ---------------------------------------------------------------------------
# Series fusion
# ---------------------------------------------------------------------------


def fuse_frames(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-frame evidence into one object-level verdict.

    Rules, in order:

    1. Positive evidence beats absence: a confident 100% frame cannot be
       cancelled by an empty frame (the drone simply did not see the house in
       that shot).
    2. A 100% claim needs at least one frame above the auto threshold, or a
       clear majority of weaker 100% frames.
    3. Cross-frame agreement raises confidence; contradiction lowers it.
    4. If nothing is confident enough, the object is sent to manual review
       instead of being labelled 0%.
    """
    usable = [f for f in frames if f]
    if not usable:
        return {
            "stage": STAGE_0,
            "confidence": 0.0,
            "frames": 0,
            "agreement": 0.0,
            "needs_manual_review": True,
            "reason": "no_frames",
            "evidence": {},
        }

    total = len(usable)
    by_stage: dict[str, list[float]] = {STAGE_0: [], STAGE_50: [], STAGE_100: []}
    for frame in usable:
        by_stage.setdefault(frame["stage"], []).append(float(frame["confidence"]))

    max_100 = max(by_stage[STAGE_100], default=0.0)
    max_50 = max(by_stage[STAGE_50], default=0.0)
    max_0 = max(by_stage[STAGE_0], default=0.0)
    n_100 = len(by_stage[STAGE_100])
    n_50 = len(by_stage[STAGE_50])
    n_0 = len(by_stage[STAGE_0])
    strong_100 = [c for c in by_stage[STAGE_100] if c >= MANUAL_REVIEW_THRESHOLD]

    evidence = {
        "frames_total": total,
        "frames_100": n_100,
        "frames_50": n_50,
        "frames_0": n_0,
        "max_confidence_100": round(max_100, 3),
        "max_confidence_50": round(max_50, 3),
        "max_confidence_0": round(max_0, 3),
        "strong_100_frames": len(strong_100),
    }

    def _agreement(stage: str) -> float:
        count = len(by_stage[stage])
        return round(count / total, 3) if total else 0.0

    # 100 % подтверждается автоматически, только если его не перевешивает
    # свидетельство «незавершённой» стадии. На стройплощадке с поддонами один
    # кадр иногда ловит соседскую кровлю, но три кадра из пяти видят фундамент
    # и стены — доверять одному кадру против трёх нельзя.
    if strong_100 and n_50 <= n_100:
        agreement = _agreement(STAGE_100)
        # Бонус за согласие имеет смысл только когда кадров больше одного:
        # у одиночного снимка «согласованность» всегда 100%, но это не факт.
        agreement_bonus = 0.10 * agreement if total > 1 else 0.0
        confidence = min(0.98, max_100 + agreement_bonus + 0.02 * min(len(strong_100) - 1, 3))
        return {
            "stage": STAGE_100,
            "confidence": round(confidence, 3),
            "frames": total,
            "agreement": agreement,
            "needs_manual_review": False,
            "reason": "series_roof_confirmed",
            "evidence": evidence,
        }

    # No single frame is auto-confident, but several frames still point to a
    # finished roof (weak detections of a real house from a bad angle).
    if n_100 >= 2 and n_100 >= max(n_50, n_0):
        agreement = _agreement(STAGE_100)
        confidence = round(min(0.74, max_100 + 0.05), 3)
        return {
            "stage": STAGE_100,
            "confidence": confidence,
            "frames": total,
            "agreement": agreement,
            "needs_manual_review": True,
            "reason": "series_weak_roof_consensus",
            "evidence": evidence,
        }

    # Один кадр с кровлей, но уверенность ниже порога авто-решения. Раньше
    # такой кадр проваливался в ветку «0 %» - и владелец загружавший ОДНО фото
    # готового дома получал «ничего не найдено» и повторный вылет. Это неверно:
    # детектор кровлю нашёл, значит стадия не 0 %. Ставим 100 % и отправляем
    # на подтверждение человеку.
    if n_100 > 0 and n_100 >= n_50 and n_100 >= n_0:
        agreement = _agreement(STAGE_100)
        return {
            "stage": STAGE_100,
            "confidence": round(max_100, 3),
            "frames": total,
            "agreement": agreement,
            "needs_manual_review": max_100 < MANUAL_REVIEW_THRESHOLD,
            "reason": "single_roof_frame_needs_review",
            "evidence": evidence,
        }

    # A weak "50%" claim is still better evidence than "nothing seen": the
    # detector did localise a foundation or walls, it was just not confident.
    if n_50 > 0 and n_50 >= n_0 and max_50 >= 0.30:
        agreement = _agreement(STAGE_50)
        confidence = round(min(0.74, max_50 + 0.03 * agreement), 3)
        return {
            "stage": STAGE_50,
            "confidence": confidence,
            "frames": total,
            "agreement": agreement,
            "needs_manual_review": True,
            "reason": "series_unfinished_consensus",
            "evidence": evidence,
        }

    # Only "nothing found" across the whole series. Absence of evidence, so
    # confidence stays low unless every frame agrees.
    agreement = _agreement(STAGE_0)
    confidence = round(min(0.60, max_0 * (0.75 + 0.25 * agreement)), 3)
    return {
        "stage": STAGE_0,
        "confidence": confidence,
        "frames": total,
        "agreement": agreement,
        "needs_manual_review": confidence < MANUAL_REVIEW_THRESHOLD,
        "reason": "series_no_object_found",
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Public entry point used by the app / CLI
# ---------------------------------------------------------------------------


def analyze_series(
    sources: Iterable[Any],
    conf_threshold: float = 0.25,
    *,
    use_zoom: bool = True,
    max_frames: int = 24,
) -> dict[str, Any] | None:
    """Analyse a flight series and return the fusion record.

    Returns ``None`` when the ONNX model is missing, so callers can fall back
    to the legacy OpenCV path.

    ``max_frames`` keeps the Streamlit interaction snappy: one frame costs
    ~0.45 s on CPU (full pass + zoom pass), so the first 24 frames of a
    flight are analysed and the rest are reported as not analysed.
    """
    if not model_available():
        return None

    frame_reports: list[dict[str, Any]] = []
    names: list[str] = []
    for index, source in enumerate(sources):
        if index >= max_frames:
            break
        report = analyze_frame(source, conf_threshold=conf_threshold, use_zoom=use_zoom)
        if report is None:
            return None
        name = getattr(source, "name", None) or (Path(source).name if isinstance(source, (str, os.PathLike)) else f"frame_{index}")
        names.append(str(name))
        frame_reports.append(report)

    fused = fuse_frames(frame_reports)
    classes: set[str] = set()
    for report in frame_reports:
        classes.update(report.get("classes") or [])
    ordered = [name for name in CLASS_NAMES if name in classes]

    best_frame_index = max(
        range(len(frame_reports)),
        key=lambda i: (frame_reports[i]["confidence"], frame_reports[i]["stage"] == fused["stage"]),
    ) if frame_reports else 0

    return {
        **fused,
        "detected_classes": ordered,
        "frames": [
            {
                "name": names[i],
                "stage": frame_reports[i]["stage"],
                "confidence": frame_reports[i]["confidence"],
                "classes": frame_reports[i]["classes"],
                "reason": frame_reports[i]["reason"],
                "object_box": frame_reports[i].get("object_box"),
            }
            for i in range(len(frame_reports))
        ],
        "best_frame": names[best_frame_index] if names else None,
        "frames_analyzed": len(frame_reports),
        "model": "sberstroi_yolov8n.onnx",
    }


# ---------------------------------------------------------------------------
# Overlay for human verification
# ---------------------------------------------------------------------------

_CLASS_COLORS = {
    "roof_finished": (0, 200, 0),
    "walls": (0, 150, 255),
    "foundation": (200, 120, 0),
    "windows": (255, 200, 0),
    "doors": (255, 0, 255),
    "fence": (128, 128, 128),
}


def render_overlay(
    source: Any,
    frame: dict[str, Any],
    *,
    stage: str | None = None,
    confidence: float | None = None,
    max_side: int = 1600,
) -> np.ndarray | None:
    """Draw detections and the object of interest on a copy of the frame.

    The result is what a human uses to check the model: green boxes are the
    object of interest, grey boxes are background detections (neighbours,
    fences) that were **not** used for the decision.
    """
    image = decode_image(source)
    if image is None:
        return None

    height, width = image.shape[:2]
    canvas = image.copy()
    object_box = frame.get("object_box")
    object_boxes = {tuple(object_box)} if object_box else set()

    for detection in frame.get("detections") or []:
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        is_object = tuple(detection["box"]) in object_boxes
        color = _CLASS_COLORS.get(detection["class_name"], (255, 255, 255))
        thickness = max(3, int(min(width, height) / 220)) if is_object else max(1, int(min(width, height) / 400))
        if not is_object:
            color = tuple(int(c * 0.55 + 60) for c in color)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        font_scale = max(0.5, min(width, height) / 1400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            canvas, label, (x1 + 3, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA,
        )

    # Centre window used to decide "is this the pledged object".
    cx0, cx1 = int(width * (0.5 - CENTRAL_X)), int(width * (0.5 + CENTRAL_X))
    cy0, cy1 = int(height * (0.5 - CENTRAL_Y)), int(height * (0.5 + CENTRAL_Y))
    cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), (255, 255, 255), max(1, int(min(width, height) / 500)))

    header = f"stage {stage or frame.get('stage')}  conf {confidence if confidence is not None else frame.get('confidence')}"
    cv2.rectangle(canvas, (0, 0), (max(560, int(width * 0.45)), int(height * 0.055) + 8), (0, 0, 0), -1)
    cv2.putText(
        canvas, header, (10, int(height * 0.045)),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.8, min(width, height) / 900), (0, 255, 0), 2, cv2.LINE_AA,
    )

    scale = max_side / max(width, height)
    if scale < 1.0:
        canvas = cv2.resize(canvas, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return canvas


def save_image(path: str | os.PathLike[str], image: np.ndarray, quality: int = 90) -> None:
    """Write an image to disk safely on Windows (Cyrillic paths included)."""
    suffix = Path(path).suffix or ".jpg"
    ok, buffer = cv2.imencode(suffix, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError(f"could not encode image for {path}")
    Path(path).write_bytes(buffer.tobytes())


def to_display_rgb(source: Any, max_side: int = 1600) -> np.ndarray | None:
    """Prepare any image for the UI as a plain 8-bit RGB array.

    Why not pass the raw bytes to ``st.image``: Streamlit re-encodes the payload
    with Pillow, and Pillow cannot write 16-bit images (mode ``I;16``) as JPEG.
    The drone's multispectral channels (``_MS_*.TIF``) are exactly that, so the
    preview used to crash with ``OSError: cannot write mode I;16 as JPEG`` while
    the analysis itself ran fine. Decoding through OpenCV normalises any depth
    and channel count to 8-bit BGR first.
    """
    image = decode_image(source)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return rgb


__all__ = [
    "analyze_frame",
    "analyze_series",
    "classical_cues",
    "decode_image",
    "fuse_frames",
    "render_overlay",
    "save_image",
    "to_display_rgb",
]

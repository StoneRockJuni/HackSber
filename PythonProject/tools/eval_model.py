r"""Честная оценка нейросети: детекция + стадия + доля отказов.

Зачем: «проверить нейронку картинками из интернета» можно, но только как
out-of-distribution тест, а не как измерение точности. Чтобы получить цифру,
нужен размеченный набор. Этот скрипт умеет работать с двумя форматами:

1. **Папка в формате YOLO** (images/ + labels/). Так размечен демо-датасет:

       .\.venv\Scripts\python.exe tools\eval_model.py --yolo ..\SberStroi_dataset\yolo\images\val ^
           --labels ..\SberStroi_dataset\yolo\labels\val

2. **CSV-манифест** для своих фото (в том числе найденных в интернете):

       path,true_stage,series
       photos/house1.jpg,100,obj-A
       photos/empty1.jpg,0,obj-B

       .\.venv\Scripts\python.exe tools\eval_model.py --manifest my_test.csv

Что считается:

* **детекция**: precision / recall / F1 / AP50 по каждому классу при IoU 0.5;
* **стадия**: матрица ошибок 0/50/100, accuracy, accuracy только на тех
  кадрах, где модель решилась сама (confidence >= 0.75), и доля отказов
  (abstention) - именно она показывает цену «не врать»;
* **серия**: если задан `series`, кадры объединяются тем же fusion, что и в
  приложении, и сравниваются с истинной стадией объекта.

Скрипт ничего не переобучает и не подкручивает пороги под тест-набор.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photo_ai import analyze_frame, fuse_frames  # noqa: E402
from sber_vision import CLASS_NAMES, MANUAL_REVIEW_THRESHOLD, model_available, stage_from_classes  # noqa: E402

STAGES = ("0%", "50%", "100%")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _average_precision(records: list[tuple[float, bool]], total_gt: int) -> float:
    """All-point interpolated AP for one class."""
    if total_gt == 0 or not records:
        return 0.0
    records.sort(key=lambda item: item[0], reverse=True)
    tp = fp = 0
    precisions, recalls = [], []
    for _, is_true in records:
        if is_true:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)
    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * (recall - previous_recall)
        previous_recall = recall
    return ap


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(value) for value in parts[1:5])
        boxes.append(
            {
                "class_name": CLASS_NAMES[class_id],
                "box": [
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ],
            }
        )
    return boxes


def evaluate_detection(
    samples: list[dict[str, Any]], iou_threshold: float = 0.5
) -> dict[str, Any]:
    """Greedy 1:1 matching per image -> precision/recall/AP per class."""
    per_class: dict[str, dict[str, list[tuple[float, bool]]]] = defaultdict(lambda: {"records": []})
    gt_totals: Counter[str] = Counter()
    pred_totals: Counter[str] = Counter()

    for sample in samples:
        truth = sample.get("gt_boxes") or []
        predicted = sample.get("pred_boxes") or []
        for box in truth:
            gt_totals[box["class_name"]] += 1
        matched_gt: set[int] = set()
        for prediction in sorted(predicted, key=lambda d: d["confidence"], reverse=True):
            best_index, best_iou = None, 0.0
            for index, ground in enumerate(truth):
                if index in matched_gt or ground["class_name"] != prediction["class_name"]:
                    continue
                value = iou(prediction["box"], ground["box"])
                if value > best_iou:
                    best_index, best_iou = index, value
            is_true = best_index is not None and best_iou >= iou_threshold
            if is_true:
                matched_gt.add(int(best_index))
            pred_totals[prediction["class_name"]] += 1
            per_class[prediction["class_name"]]["records"].append((prediction["confidence"], is_true))

    result: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        records = per_class.get(class_name, {}).get("records", [])
        tp = sum(1 for _, ok in records if ok)
        fp = sum(1 for _, ok in records if not ok)
        fn = max(0, gt_totals[class_name] - tp)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        result[class_name] = {
            "gt": gt_totals[class_name],
            "predictions": pred_totals[class_name],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "ap50": round(_average_precision(records, gt_totals[class_name]), 3) if records else 0.0,
        }
    return result


def evaluate_stage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    decided = 0
    decided_correct = 0
    for sample in samples:
        true_stage = sample["true_stage"]
        predicted_stage = sample["pred_stage"]
        confusion[(true_stage, predicted_stage)] += 1
        correct += int(true_stage == predicted_stage)
        if not sample.get("needs_review"):
            decided += 1
            decided_correct += int(true_stage == predicted_stage)
    total = len(samples)
    return {
        "frames": total,
        "stage_accuracy": round(correct / total, 3) if total else None,
        "auto_decided": decided,
        "abstention_rate": round(1 - decided / total, 3) if total else None,
        "stage_accuracy_when_decided": round(decided_correct / decided, 3) if decided else None,
        "confusion": {f"{true}->{pred}": count for (true, pred), count in sorted(confusion.items())},
    }


def evaluate_series(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["series"]].append(sample)
    rows = []
    correct = 0
    for series, items in sorted(grouped.items()):
        fused = fuse_frames([item["frame"] for item in items])
        true_stages = Counter(item["true_stage"] for item in items)
        true_stage = true_stages.most_common(1)[0][0]
        correct += int(true_stage == fused["stage"])
        rows.append(
            {
                "series": series,
                "frames": len(items),
                "true_stage": true_stage,
                "predicted_stage": fused["stage"],
                "confidence": fused["confidence"],
                "needs_review": fused["needs_manual_review"],
                "reason": fused["reason"],
            }
        )
    return {
        "series_count": len(rows),
        "series_accuracy": round(correct / len(rows), 3) if rows else None,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_yolo_set(images_dir: Path, labels_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    problems: list[str] = []
    for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        label_path = labels_dir / f"{image_path.stem}.txt"
        frame = analyze_frame(image_path)
        if frame is None or not frame["detections"]:
            if frame is None:
                problems.append(f"модель недоступна: {image_path.name}")
            else:
                problems.append(f"нет детекций: {image_path.name}")
        height = width = 0
        try:
            import cv2
            import numpy as np

            raw = np.fromfile(str(image_path), dtype=np.uint8)
            decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            height, width = decoded.shape[:2]
        except Exception:
            problems.append(f"не удалось прочитать размер: {image_path.name}")
        gt_boxes = read_yolo_labels(label_path, width, height) if label_path.exists() else []
        true_classes = {box["class_name"] for box in gt_boxes}
        true_stage, _ = stage_from_classes(true_classes)
        samples.append(
            {
                "name": image_path.name,
                "series": image_path.name[:18],
                "gt_boxes": gt_boxes,
                "pred_boxes": (frame or {}).get("detections", []),
                "true_stage": true_stage,
                "pred_stage": (frame or {}).get("stage", "0%"),
                "needs_review": bool((frame or {}).get("needs_manual_review", True)),
                "frame": frame,
            }
        )
    return samples, problems


def load_manifest(manifest_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    problems: list[str] = []
    base = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = (base / path).resolve()
            if not path.exists():
                problems.append(f"нет файла: {path}")
                continue
            frame = analyze_frame(path)
            if frame is None:
                problems.append("модель недоступна")
                continue
            true_stage = str(row.get("true_stage") or "").strip()
            if true_stage not in STAGES:
                problems.append(f"{path.name}: true_stage должен быть 0%, 50% или 100%")
                continue
            samples.append(
                {
                    "name": path.name,
                    "series": (row.get("series") or path.stem).strip(),
                    "gt_boxes": [],
                    "pred_boxes": frame["detections"],
                    "true_stage": true_stage,
                    "pred_stage": frame["stage"],
                    "needs_review": frame["needs_manual_review"],
                    "frame": frame,
                }
            )
    return samples, problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Честная оценка модели СтройКонтур")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--yolo", help="Папка с изображениями YOLO-набора")
    source.add_argument("--manifest", help="CSV: path,true_stage[,series]")
    parser.add_argument("--labels", help="Папка с YOLO-разметкой (для --yolo)")
    parser.add_argument("--manifest-out", default="", help="Куда записать JSON-отчёт")
    parser.add_argument("--iou", type=float, default=0.5, help="Порог IoU для детекции")
    args = parser.parse_args()

    if not model_available():
        print("ОШИБКА: ONNX-модель не найдена (models/sberstroi_yolov8n.onnx)", file=sys.stderr)
        return 1

    if args.yolo:
        if not args.labels:
            parser.error("--labels обязателен вместе с --yolo")
        samples, problems = load_yolo_set(Path(args.yolo), Path(args.labels))
    else:
        samples, problems = load_manifest(Path(args.manifest))

    if not samples:
        print("Нет ни одного пригодного примера.", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    with_gt = [sample for sample in samples if sample["gt_boxes"]]
    stage_report = evaluate_stage(samples)
    series_report = evaluate_series(samples)
    report: dict[str, Any] = {
        "model": "models/sberstroi_yolov8n.onnx",
        "samples": len(samples),
        "detection": evaluate_detection(with_gt, iou_threshold=args.iou) if with_gt else None,
        "stage": stage_report,
        "series": series_report,
        "problems": problems,
        "warning": (
            "Демо-набор мал (десятки кадров, 2 объекта). Это не промышленный бенчмарк. "
            "Фото из интернета вне этого набора - out-of-distribution: они проверяют "
            "устойчивость, но не заменяют слепую полевую валидацию."
        ),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.manifest_out:
        Path(args.manifest_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nотчёт сохранён: {args.manifest_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

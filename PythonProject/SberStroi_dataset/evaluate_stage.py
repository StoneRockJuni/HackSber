"""Evaluate the exported ONNX model as a stage classifier on the val split.

This is a small, honest stage-level check, not a production benchmark:

    .\\.venv\\Scripts\\python.exe SberStroi_dataset\\evaluate_stage.py

It reads val labels, computes ground-truth 0/50/100% with the same conservative
mapping used by the app, runs the ONNX model via `sber_vision`, and writes
`models/sberstroi_stage_eval.json`.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sber_vision import CLASS_NAMES, analyze_construction, model_available, stage_from_classes  # noqa: E402

VAL_IMAGES = ROOT / "SberStroi_dataset" / "yolo" / "images" / "val"
VAL_LABELS = ROOT / "SberStroi_dataset" / "yolo" / "labels" / "val"
OUT = Path(os.getenv("STAGE_EVAL_OUT", str(ROOT / "models" / "sberstroi_stage_eval.json")))


def true_classes(label_path: Path) -> set[str]:
    result: set[str] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(CLASS_NAMES[int(line.split()[0])])
    return result


def main() -> None:
    if not model_available():
        raise SystemExit("ONNX model not found; run SberStroi_dataset/train_yolo.py first")

    rows = []
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    for image in sorted(VAL_IMAGES.glob("*.jpg")):
        label = VAL_LABELS / f"{image.stem}.txt"
        ground_truth_classes = true_classes(label)
        true_stage, _ = stage_from_classes(ground_truth_classes)
        analysis = analyze_construction(image)
        if analysis is None:
            raise RuntimeError("model disappeared during evaluation")
        predicted_stage = analysis["construction_analysis"]["detected_stage"]
        confidence = analysis["construction_analysis"]["confidence_stage"]
        confusion[(true_stage, predicted_stage)] += 1
        correct += int(true_stage == predicted_stage)
        rows.append(
            {
                "image": image.name,
                "true_classes": sorted(ground_truth_classes),
                "true_stage": true_stage,
                "predicted_classes": analysis["construction_analysis"]["detected_classes"],
                "predicted_stage": predicted_stage,
                "confidence": confidence,
                "needs_manual_review": analysis["construction_analysis"]["needs_manual_review"],
            }
        )

    total = len(rows)
    report = {
        "val_images": total,
        "stage_accuracy": round(correct / total, 4) if total else None,
        "confusion": {f"{true}->{pred}": count for (true, pred), count in sorted(confusion.items())},
        "rows": rows,
        "warning": (
            "10 val images from 2 flight sequences. This measures the demo, "
            "not production accuracy or arbitrary internet photos."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved:", OUT)


if __name__ == "__main__":
    main()

"""Export the best holdout checkpoint and its honest metrics.

The first training run may still be using the last-epoch weights for export
depending on the Ultralytics version. This script always reloads `best.pt`,
evaluates it on the sequence-level val split, and writes:

- `models/sberstroi_yolov8n_holdout.onnx`
- `models/sberstroi_yolov8n_holdout.json`

These are the artifacts to cite for generalisation. The app-facing ONNX may be
the demo-fit model from `train_yolo_all.py`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "SberStroi_dataset" / "yolo" / "data.yaml"
BEST = ROOT / "SberStroi_dataset" / "runs" / "sberstroi_yolov8n" / "weights" / "best.pt"
MODELS = ROOT / "models"
CLASS_NAMES = ["doors", "fence", "foundation", "roof_finished", "walls", "windows"]


def main() -> None:
    if not BEST.is_file():
        raise FileNotFoundError(f"best.pt not found: {BEST}")
    MODELS.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(BEST))
    metrics = model.val(data=str(DATA), imgsz=640, device="cpu", workers=0, plots=False)
    box = metrics.box
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "holdout_sequence_split",
        "dataset": "SberStroi dataset, 40 train / 10 val by whole flight sequence",
        "train_images": 40,
        "val_images": 10,
        "epochs": 50,
        "imgsz": 640,
        "batch": 8,
        "classes": CLASS_NAMES,
        "metrics": {
            "mAP50-95": round(float(box.map), 4),
            "mAP50": round(float(box.map50), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "per_class_mAP50-95": {
                name: round(float(value), 4)
                for name, value in zip(CLASS_NAMES, box.maps.tolist())
            },
        },
        "warning": (
            "Tiny 40/10 demo dataset. This is the only metric that can be "
            "called a held-out generalisation check, and even it is not "
            "production evidence for arbitrary internet photos."
        ),
    }

    onnx_path = Path(model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False))
    target = MODELS / "sberstroi_yolov8n_holdout.onnx"
    shutil.copy2(onnx_path, target)
    metadata["onnx_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata["onnx_bytes"] = target.stat().st_size
    (MODELS / "sberstroi_yolov8n_holdout.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("model:", target)


if __name__ == "__main__":
    main()

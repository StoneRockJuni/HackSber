"""Fine-tune the held-out model on all 50 demo images for the live demo.

Why this exists
---------------
The honest sequence-level split (`train_yolo.py`) leaves two whole flight
sequences out. That is the right way to measure generalisation, but with only
40 training images the resulting model is conservative and often asks for
manual review.

For the hackathon stand we additionally fit a **demo model** on all 50 images.
This is intentionally not a benchmark: the 10 val images are now also in train.
Keep the holdout model and its metrics for any claim about generalisation.

Run after `train_yolo.py`:

    C:\\Users\\Blackview\\AppData\\Local\\Temp\\stroykontur_ml_venv\\Scripts\\python.exe ^
        SberStroi_dataset\\train_yolo_all.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "SberStroi_dataset" / "all_data.yaml"
RUNS = ROOT / "SberStroi_dataset" / "runs"
MODELS = ROOT / "models"
HOLDOUT_BEST = RUNS / "sberstroi_yolov8n" / "weights" / "best.pt"
HOLDOUT_ONNX = MODELS / "sberstroi_yolov8n.onnx"
CLASS_NAMES = ["doors", "fence", "foundation", "roof_finished", "walls", "windows"]


def main() -> None:
    if not HOLDOUT_BEST.is_file():
        raise FileNotFoundError(f"run train_yolo.py first: {HOLDOUT_BEST}")
    MODELS.mkdir(parents=True, exist_ok=True)

    # Preserve the honest sequence-split artifact and metrics before replacing
    # the app-facing ONNX with the demo-fit model.
    if HOLDOUT_ONNX.is_file():
        shutil.copy2(HOLDOUT_ONNX, MODELS / "sberstroi_yolov8n_holdout.onnx")
    metrics_json = MODELS / "sberstroi_yolov8n.json"
    if metrics_json.is_file():
        shutil.copy2(metrics_json, MODELS / "sberstroi_yolov8n_holdout.json")

    model = YOLO(str(HOLDOUT_BEST))
    result = model.train(
        data=str(DATA),
        epochs=20,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        patience=10,
        seed=42,
        project=str(RUNS),
        name="sberstroi_all",
        exist_ok=True,
        plots=True,
        verbose=True,
        # Fine-tuning on 50 images: keep the model close to the pretrained
        # detector and avoid aggressive mosaic changes.
        mosaic=0.0,
        close_mosaic=0,
        lr0=0.001,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"best.pt not found: {best}")

    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(DATA), imgsz=640, device="cpu", workers=0, plots=False)
    box = metrics.box
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "demo_fit",
        "warning": (
            "All 50 images were used for training; the 10 val images are not "
            "held out. Do not present these metrics as generalisation. "
            "Use sberstroi_yolov8n_holdout.json for the honest sequence split."
        ),
        "base_checkpoint": str(HOLDOUT_BEST.relative_to(ROOT)),
        "epochs": 20,
        "imgsz": 640,
        "batch": 8,
        "classes": CLASS_NAMES,
        "fit_metrics_NOT_generalisation": {
            "mAP50-95": round(float(box.map), 4),
            "mAP50": round(float(box.map50), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        },
    }

    onnx_path = Path(best_model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False))
    target = MODELS / "sberstroi_yolov8n.onnx"
    shutil.copy2(onnx_path, target)
    metadata["onnx_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata["onnx_bytes"] = target.stat().st_size
    (MODELS / "sberstroi_yolov8n.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("model:", target)


if __name__ == "__main__":
    main()

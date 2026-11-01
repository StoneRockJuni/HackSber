"""Train the SberStroi YOLOv8n detector and export ONNX for the app.

Run from the project root with the training environment (Ultralytics + torch):

    C:\\Users\\Blackview\\AppData\\Local\\Temp\\stroykontur_ml_venv\\Scripts\\python.exe ^
        SberStroi_dataset\\train_yolo.py

The exported model is copied to `models/sberstroi_yolov8n.onnx`; the Streamlit
app loads it with OpenCV DNN and does not need torch at runtime.

Important: 40 training images / 10 validation images is a hackathon demo
dataset, not a production benchmark. The split is by whole flight sequence to
avoid near-duplicate leakage.
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
RUNS = ROOT / "SberStroi_dataset" / "runs"
MODELS = ROOT / "models"
CLASS_NAMES = ["doors", "fence", "foundation", "roof_finished", "walls", "windows"]


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    result = model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        patience=15,
        seed=42,
        project=str(RUNS),
        name="sberstroi_yolov8n",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"best.pt not found: {best}")

    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(DATA), imgsz=640, device="cpu", workers=0, plots=False)
    box = metrics.box
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": "yolov8n.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 8,
        "dataset": "SberStroi dataset (Roboflow export), sequence-level split",
        "train_images": 40,
        "val_images": 10,
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
            "Tiny demo dataset; metrics are not a production benchmark. "
            "Arbitrary internet photos are out of distribution."
        ),
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

"""Prepare a leakage-free YOLO split from the SberStroi Roboflow export.

The original notebook copied 10 random images from train to val. Because the
dataset contains 5 near-duplicate frames per flight sequence, that split leaks
the same object into train and val and inflates metrics.

This script holds out whole flight sequences instead. It keeps the demo small
and reproducible:

    .\\.venv\\Scripts\\python.exe SberStroi_dataset\\prepare_yolo_split.py

The result is written to `SberStroi_dataset/yolo/` and is used for training
only. Runtime inference uses the exported ONNX model (if present) or the
existing OpenCV fallback.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_IMAGES = ROOT / "train" / "images"
SRC_LABELS = ROOT / "train" / "labels"
OUT = ROOT / "yolo"

# Whole sequences (5 frames each). One foundation-only sequence and one
# roof+doors sequence give us both 50% and 100% cases in validation.
VAL_SEQUENCES = {
    "DJI_20260807142835_0002_D_JPG",  # foundation -> 50%
    "DJI_20260807144205_0005_D_JPG",  # roof + doors + windows + walls -> 100%
}

CLASS_NAMES = ["doors", "fence", "foundation", "roof_finished", "walls", "windows"]


def sequence_of(image_name: str) -> str:
    return image_name.split(".rf.")[0]


def main() -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            folder = OUT / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob("*"):
                old.unlink()

    counts = {"train": Counter(), "val": Counter()}
    for image in sorted(SRC_IMAGES.glob("*.jpg")):
        split = "val" if sequence_of(image.name) in VAL_SEQUENCES else "train"
        label = SRC_LABELS / f"{image.stem}.txt"
        if not label.exists():
            print(f"skip without label: {image.name}")
            continue
        shutil.copy2(image, OUT / "images" / split / image.name)
        shutil.copy2(label, OUT / "labels" / split / label.name)
        counts[split]["images"] += 1
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[split][CLASS_NAMES[int(line.split()[0])]] += 1

    (OUT / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {OUT.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {CLASS_NAMES!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("prepared:", OUT)
    for split in ("train", "val"):
        print(split, dict(counts[split]))


if __name__ == "__main__":
    main()

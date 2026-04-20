"""
STEP 10: Retrain with improved dataset
- Resumes from previous best.pt OR starts fresh with same base model
- Tracks version number in run name (v1, v2, v3...)
"""

from ultralytics import YOLO
import torch
import os
import glob

DATA_YAML   = "data/marine_detection.yaml"
PROJECT_DIR = "runs/marine"
BASE_MODEL  = "yolov8s.pt"

# ─── CONFIG ─────────────────────────────────────────────────────────────────
RESUME_FROM_LAST = True    # True = resume from best previous run
                           # False = start fresh from base model

TRAIN_CONFIG = {
    "data":      DATA_YAML,
    "epochs":    100,
    "patience":  20,
    "imgsz":     640,
    "batch":     16,
    "workers":   4,
    "device":    0 if torch.cuda.is_available() else "cpu",
    "project":   PROJECT_DIR,
    "optimizer": "AdamW",
    "lr0":       0.0005,   # Lower LR for fine-tuning on improved data
    "lrf":       0.01,
    "warmup_epochs": 2,
    "cos_lr":    True,
    "augment":   True,
    "hsv_h":     0.015,
    "hsv_s":     0.7,
    "hsv_v":     0.4,
    "degrees":   10.0,
    "translate": 0.1,
    "scale":     0.5,
    "flipud":    0.1,
    "fliplr":    0.5,
    "mosaic":    1.0,
    "mixup":     0.1,
    "copy_paste": 0.1,
    "save":      True,
    "save_period": 10,
    "val":       True,
    "plots":     True,
    "verbose":   True,
}
# ────────────────────────────────────────────────────────────────────────────


def get_next_version():
    """Auto-detect next run version number."""
    existing = glob.glob(os.path.join(PROJECT_DIR, "v*"))
    versions = []
    for e in existing:
        name = os.path.basename(e)
        if name.startswith("v") and name[1:].isdigit():
            versions.append(int(name[1:]))
    return max(versions, default=0) + 1


def find_best_weights():
    """Find the best.pt from the most recent run."""
    existing = glob.glob(os.path.join(PROJECT_DIR, "v*", "weights", "best.pt"))
    if not existing:
        return None
    # Sort by version number
    def version_num(p):
        parts = p.split(os.sep)
        for part in parts:
            if part.startswith("v") and part[1:].isdigit():
                return int(part[1:])
        return 0
    existing.sort(key=version_num, reverse=True)
    return existing[0]


def retrain():
    version = get_next_version()
    run_name = f"v{version}"
    print(f"Starting run: {run_name}")

    if RESUME_FROM_LAST:
        best_pt = find_best_weights()
        if best_pt:
            print(f"Resuming from: {best_pt}")
            model = YOLO(best_pt)
        else:
            print(f"No previous weights found. Starting fresh from {BASE_MODEL}")
            model = YOLO(BASE_MODEL)
    else:
        print(f"Starting fresh from {BASE_MODEL}")
        model = YOLO(BASE_MODEL)

    TRAIN_CONFIG["name"] = run_name
    TRAIN_CONFIG["exist_ok"] = False

    results = model.train(**TRAIN_CONFIG)

    best_path = os.path.join(PROJECT_DIR, run_name, "weights", "best.pt")
    print(f"\nBest weights: {best_path}")
    return results


if __name__ == "__main__":
    retrain()
    print("\n[DONE] Step 10 complete.")

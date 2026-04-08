import sys
import os
import numpy as np
import torch

from model_loader import load_model
from depth.depth_estimator import estimate_depth
from analysis.water_quality import analyze_water_quality

def verify():
    print("1. Testing Model Loader (Model structure mismatch)")
    model = load_model()
    if model is not None:
        print("✅ Enhancement model loaded successfully.")
    else:
        print("❌ Enhancement model failed to load!")

    print("\n2. Testing Water Quality (Turbidity UDCP Bug)")
    # Create fake underwater image where red channel is nearly zero
    fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
    fake_img[:, :, 1] = 60  # Green
    fake_img[:, :, 2] = 120 # Blue
    # Since red is 0, old DCP would say 0 haze (Clear).
    # New UDCP looks at green and blue minimums (60), which means dark = ~0.23, which is Moderate/High.
    wq = analyze_water_quality(fake_img)
    print("Water Quality Analysis Output:")
    print(wq)
    if wq["turbidity_level"] != "Clear":
        print("✅ Water quality successfully avoided the 'always clear' red channel bug!")
    else:
        print("❌ Water quality might still be ignoring underwater physics.")

    print("\n3. Testing Depth Estimator (Keyword arg bug)")
    try:
        depth = estimate_depth(fake_img)
        if "status" in depth and depth["status"] == "ok":
            print("✅ Depth estimator ran without keyword errors!")
        else:
            print("❌ Depth estimator returned error:", depth.get("error"))
    except Exception as e:
        print("❌ Depth estimator crash:", str(e))

if __name__ == "__main__":
    verify()

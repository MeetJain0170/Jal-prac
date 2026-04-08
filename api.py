"""
api.py — JalDrishti Maritime Security Vision System  (Flask backend)
=====================================================================

Endpoints
---------
  POST /api/enhance        — U-Net image enhancement + quality metrics
  POST /api/detect         — YOLO maritime object detection
  POST /api/depth          — MiDaS monocular depth estimation
  POST /api/analyze-water  — Water turbidity / visibility analysis
  POST /api/process-video  — Video enhancement + detection pipeline
  GET  /api/status         — Model / system health check
"""

from __future__ import annotations

import io
import os
import base64
import time

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

import config
from train.model import UNet, count_parameters
from enhance import enhance_image, enhance_opencv, calculate_metrics_full
from model_loader import load_model as _shared_load_model

# ── Analysis modules ─────────────────────────────────────────────────────────
from analysis.water_quality  import analyze_water_quality
from analysis.threat_analysis import compute_threat_score
from analysis.heatmap        import heatmap_to_base64, enhancement_stats
from analysis.image_quality  import compute_all_metrics

# ── Detection & depth (lazy imports inside routes to avoid hard crash) ────────


app  = Flask(__name__, static_folder="static")
CORS(app)

_MODEL:    UNet | None = None
_DETECTOR = None
_DEPTH     = None


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> UNet | None:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    m = _shared_load_model()
    _MODEL = m
    return _MODEL


def _load_detector():
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    try:
        from detection.yolo_detector import get_detector
        _DETECTOR = get_detector(
            weights     = config.YOLO_WEIGHTS,
            conf_thresh = config.YOLO_CONF_THRESH,
            iou_thresh  = config.YOLO_IOU_THRESH,
            img_size    = config.YOLO_IMG_SIZE,
        )
        return _DETECTOR
    except Exception as exc:
        print(f"[api] detector load error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_image(file_storage) -> Image.Image:
    return Image.open(io.BytesIO(file_storage.read())).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── /api/enhance ──────────────────────────────────────────────────────────────

@app.route("/api/enhance", methods=["POST"])
def enhance():
    model = _load_model()
    if model is None:
        return jsonify({"error": "No trained model. Run: python train.py"}), 400
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        t0  = time.perf_counter()
        img = _read_image(request.files["image"])
        orig_np = np.array(img)
        w, h    = img.size

        # Dual Processing Path
        enhanced_img_hybrid = enhance_image(model, img, use_hybrid=True)
        enhanced_img_opencv = enhance_opencv(img)
        
        enh_np = np.array(enhanced_img_hybrid)
        elapsed = round(time.perf_counter() - t0, 3)

        metrics  = calculate_metrics_full(img, enhanced_img_hybrid)
        heatmap  = heatmap_to_base64(orig_np, enh_np)
        enh_stats = enhancement_stats(orig_np, enh_np)

        return jsonify({
            "success":          True,
            "enhanced_image_hybrid": _pil_to_b64(enhanced_img_hybrid),
            "enhanced_image_opencv": _pil_to_b64(enhanced_img_opencv),
            "heatmap":          heatmap,
            "size":             f"{w}x{h}",
            "device":           "GPU" if config.DEVICE.type == "cuda" else "CPU",
            "processing_time":  elapsed,
            # Metrics (Calculated against Hybrid)
            "psnr":   metrics["psnr"],
            "ssim":   metrics["ssim"],
            "eps":    metrics["eps"],
            "uiqm":   metrics["uiqm"],
            "uciqe":  metrics["uciqe"],
            # Enhancement stats
            "mean_enhancement": enh_stats["mean_enhancement"],
            "enhanced_pct":     enh_stats["enhanced_pct"],
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /api/detect ───────────────────────────────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img    = _read_image(request.files["image"])
        img_np = np.array(img)

        # Try YOLO — report gracefully if unavailable
        detector = _load_detector()
        if detector is None:
            return jsonify({
                "success":    False,
                "error":      "YOLO not available. Run: pip install ultralytics",
                "detections": [],
            }), 200

        t0 = time.perf_counter()
        detections, annotated_b64 = detector.detect_and_annotate(img_np)
        elapsed = round(time.perf_counter() - t0, 3)

        # Water context for threat scoring
        water = analyze_water_quality(img_np)
        threat = compute_threat_score(
            detections,
            turbidity_index=water.get("turbidity_index", 0.0),
            img_w=img_np.shape[1],
            img_h=img_np.shape[0],
        )

        return jsonify({
            "success":          True,
            "detections":       detections,
            "detection_count":  len(detections),
            "annotated_image":  annotated_b64,
            "threat_score":     threat["threat_score"],
            "alert_level":      threat["alert_level"],
            "recommendations":  threat["recommendations"],
            "processing_time":  elapsed,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /api/depth ────────────────────────────────────────────────────────────────

@app.route("/api/depth", methods=["POST"])
def depth():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img    = _read_image(request.files["image"])
        img_np = np.array(img)

        from depth.depth_estimator import estimate_depth
        t0     = time.perf_counter()
        result = estimate_depth(img_np)
        elapsed = round(time.perf_counter() - t0, 3)
        result["processing_time"] = elapsed
        result["success"] = result["status"] == "ok"
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /api/analyze-water ────────────────────────────────────────────────────────

@app.route("/api/analyze-water", methods=["POST"])
def analyze_water():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img    = _read_image(request.files["image"])
        img_np = np.array(img)

        t0      = time.perf_counter()
        wq      = analyze_water_quality(img_np)
        elapsed = round(time.perf_counter() - t0, 3)

        return jsonify({
            "success":                 True,
            "visibility_range_meters": wq["visibility_range_meters"],
            "turbidity_level":         wq["turbidity_level"],
            "turbidity_index":         wq["turbidity_index"],
            "contrast_loss":           wq["contrast_loss"],
            "attenuation":             wq["attenuation"],
            "water_type":              wq["water_type"],
            "processing_time":         elapsed,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /api/process-video ────────────────────────────────────────────────────────

@app.route("/api/process-video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    model = _load_model()
    if model is None:
        return jsonify({"error": "No trained model. Run: python train.py"}), 400

    try:
        from video.video_processor import process_video as _proc_vid
        from enhance import enhance_with_patches
        from PIL import Image as PILImage

        video_bytes = request.files["video"].read()
        run_det     = request.form.get("run_detection", "false").lower() == "true"
        detector    = _load_detector() if run_det else None

        def _enhance_frame(frame_np: np.ndarray) -> np.ndarray:
            pil = PILImage.fromarray(frame_np)
            out = enhance_with_patches(model, pil, patch_size=config.PATCH_SIZE,
                                        overlap=config.PATCH_OVERLAP)
            return np.array(out)

        result = _proc_vid(
            video_bytes  = video_bytes,
            enhance_fn   = _enhance_frame,
            output_dir   = config.VIDEO_OUTPUT_DIR,
            max_frames   = config.VIDEO_MAX_FRAMES,
            run_detection= run_det,
            detector     = detector,
        )

        if result["status"] != "ok":
            return jsonify({"error": result.get("error", "Processing failed")}), 500

        # Return download path (relative to server)
        rel_path = os.path.relpath(result["output_path"], start=config.BASE_DIR)
        return jsonify({
            "success":            True,
            "output_path":        rel_path,
            "frame_count":        result["frame_count"],
            "fps":                result["fps"],
            "resolution":         result["resolution"],
            "processing_time_s":  result["processing_time_s"],
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── /api/status ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    model = _load_model()
    params = count_parameters(model) if model else None

    # Check which optional modules are available
    yolo_ok  = False
    midas_ok = False
    cv2_ok   = False

    try:
        import ultralytics; yolo_ok = True
    except ImportError:
        pass
    try:
        torch.hub.list("intel-isl/MiDaS", trust_repo=True); midas_ok = True
    except Exception:
        pass
    try:
        import cv2; cv2_ok = True
    except ImportError:
        pass

    return jsonify({
        "model_loaded":  model is not None,
        "device":        str(config.DEVICE),
        "parameters":    params,
        "modules": {
            "yolo":      yolo_ok,
            "midas":     midas_ok,
            "opencv":    cv2_ok,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  JalDrishti - Maritime Security Vision System")
    print("=" * 65)
    _MODEL = _load_model()
    if _MODEL:
        print(f"  Enhancement model loaded  -> {config.DEVICE}")
        print(f"  Parameters: {count_parameters(_MODEL):,}")
    else:
        print("  WARNING: No model found - train first: python train.py")
    print(f"\n  Server: http://localhost:5500")
    print("  Endpoints:")
    for ep in ["/api/enhance", "/api/detect", "/api/depth",
               "/api/analyze-water", "/api/process-video", "/api/status"]:
        print(f"    {ep}")
    print("\n  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5500, debug=False, threaded=True)
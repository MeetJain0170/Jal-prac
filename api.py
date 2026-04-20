from __future__ import annotations

import io
import os
import json
import base64
import time
import uuid
import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

import config
from train.model import UNet, count_parameters
from enhance import enhance_image, enhance_opencv, calculate_metrics_full
from model_loader import load_model as _shared_load_model

from analysis.water_quality import analyze_water_quality
from analysis.threat_analysis import compute_threat_score

# ------------------------------------------------------------------------------
# CONFIG + LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JalAPI")

torch.set_num_threads(1)

app = Flask(__name__, static_folder="static")
CORS(app)

_MODEL: UNet | None = None
_DETECTOR = None
_DEPTH_MODEL = None


# ------------------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------------------

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        model = _shared_load_model()
        if model:
            model.to(config.DEVICE)
            model.eval()
        _MODEL = model
        return model
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None


def _load_detector():
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    try:
        if getattr(config, "ENABLE_DIVER_DETECTOR", False):
            from detection.hybrid_detector import get_detector
            _DETECTOR = get_detector(
                marine_weights=config.YOLO_WEIGHTS,
                marine_conf=config.YOLO_CONF_THRESH,
                marine_iou=config.YOLO_IOU_THRESH,
                marine_imgsz=config.YOLO_IMG_SIZE,
                diver_weights=getattr(config, "DIVER_WEIGHTS", "yolov8s.pt"),
                diver_conf=getattr(config, "DIVER_CONF_THRESH", 0.25),
                diver_iou=getattr(config, "DIVER_IOU_THRESH", 0.45),
                diver_imgsz=getattr(config, "DIVER_IMG_SIZE", config.YOLO_IMG_SIZE),
            )
        else:
            from detection.simple_detector import get_detector
            _DETECTOR = get_detector(
                weights=config.YOLO_WEIGHTS,
                conf_thresh=config.YOLO_CONF_THRESH,
                iou_thresh=config.YOLO_IOU_THRESH,
                img_size=config.YOLO_IMG_SIZE,
                tta=getattr(config, "YOLO_TTA", False),
                multi_scale=getattr(config, "YOLO_MULTI_SCALE", True),
            )
        return _DETECTOR
    except Exception as e:
        logger.warning(f"YOLO load failed: {e}")
        return None


def _load_depth():
    """Warm up the MiDaS depth model (lazy — loads on first real call)."""
    global _DEPTH_MODEL
    if _DEPTH_MODEL is not None:
        return _DEPTH_MODEL

    try:
        # depth_estimator uses an internal _load_model() so we trigger it via
        # a tiny dummy call on the module level rather than importing load_midas
        # (which never existed as a public symbol).
        from depth import depth_estimator as _de
        _DEPTH_MODEL = _de._load_model()  # returns (model, transform) tuple
        return _DEPTH_MODEL
    except Exception as e:
        logger.warning(f"Depth preload skipped (will load on first request): {e}")
        return None


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def _read_image(file):
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    return img


def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def enhancement_stats(orig, enhanced):
    diff = np.abs(orig.astype(np.float32) - enhanced.astype(np.float32))
    return {
        "mean_enhancement": float(np.mean(diff)),
        "enhanced_pct": float(np.mean(diff > 10) * 100),
    }


# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ------------------------------------------------------------------------------
# ENHANCE
# ------------------------------------------------------------------------------

@app.route("/api/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    model = _load_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        t0 = time.perf_counter()

        img = _read_image(request.files["image"])
        orig_np = np.array(img)

        fast_mode = request.form.get("fast", "false").lower() == "true"

        enhanced_hybrid = enhance_image(model, img, use_hybrid=not fast_mode)
        enhanced_op = enhance_opencv(img)

        enh_np = np.array(enhanced_hybrid)
        elapsed = round(time.perf_counter() - t0, 3)

        metrics = calculate_metrics_full(img, enhanced_hybrid)
        stats = enhancement_stats(orig_np, enh_np)

        w, h = img.size  # PIL size is (width, height)

        return jsonify({
            "success": True,
            "enhanced_image_hybrid": _pil_to_b64(enhanced_hybrid),
            "enhanced_image_opencv": _pil_to_b64(enhanced_op),
            "processing_time": elapsed,
            "size": f"{w}×{h}",
            "psnr":  metrics["psnr"],
            "ssim":  metrics["ssim"],
            "uiqm":  metrics["uiqm"],
            "uciqe": metrics["uciqe"],
            "eps":   metrics["eps"],
            **stats
        })

    except Exception as e:
        logger.error(f"Enhance failed: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# DETECT
# ------------------------------------------------------------------------------

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # `image` is the inference image (can be enhanced).
        # Optional `original` is used only for drawing UI annotations.
        infer_img = _read_image(request.files["image"])
        infer_np = np.array(infer_img)

        original_np = None
        if "original" in request.files:
            try:
                original_np = np.array(_read_image(request.files["original"]))
            except Exception:
                original_np = None

        if request.form.get("enhance_first", "false") == "true":
            model = _load_model()
            if model:
                infer_img = enhance_image(model, infer_img, use_hybrid=False)
                infer_np = np.array(infer_img)

        detector = _load_detector()
        if detector is None:
            return jsonify({"error": "YOLO unavailable"}), 200

        try:
            detections, annotated = detector.detect_and_annotate(
                infer_np,
                original_img=original_np
            )
        except Exception as det_err:
            logger.warning(f"Primary detector failed, trying fallback model: {det_err}")
            from detection.yolo_detector import get_detector
            fallback = get_detector(
                weights="yolov8s-worldv2.pt",
                conf_thresh=max(0.08, config.YOLO_CONF_THRESH),
                iou_thresh=config.YOLO_IOU_THRESH,
                img_size=min(1280, config.YOLO_IMG_SIZE),
                tta=False,
                multi_scale=True,
            )
            detections, annotated = fallback.detect_and_annotate(
                infer_np,
                original_img=original_np
            )

        water = analyze_water_quality(infer_np)
        threat = compute_threat_score(
            detections,
            turbidity_index=water.get("turbidity_index", 0),
            img_w=infer_np.shape[1],
            img_h=infer_np.shape[0],
        )

        return jsonify({
            "success": True,
            "detections": detections,
            "annotated_image": annotated,
            "threat_score": threat["threat_score"],
            "alert_level": threat["alert_level"],
            "recommendations": threat.get("recommendations", []),
            "security_objects": threat.get("security_objects", 0),
            "threat_breakdown": threat.get("breakdown", {}),
        })

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# DEPTH
# ------------------------------------------------------------------------------

@app.route("/api/depth", methods=["POST"])
def depth():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = _read_image(request.files["image"])
        img_np = np.array(img)

        t0 = time.perf_counter()

        try:
            # PRIMARY (your original working method)
            from depth.depth_estimator import estimate_depth
            result = estimate_depth(img_np)

        except Exception as e:
            print(f"[depth] primary failed: {e}")

            try:
                # FALLBACK (tensor-based)
                from depth.depth_estimator import estimate_depth_tensor

                depth_map = estimate_depth_tensor(img_np)

                result = {
                    "status": "ok",
                    "depth_map": depth_map.tolist() if hasattr(depth_map, "tolist") else None
                }

            except Exception as e2:
                print(f"[depth] fallback failed: {e2}")
                return jsonify({
                    "error": "Depth estimation failed completely",
                    "details": str(e2)
                }), 500

        elapsed = round(time.perf_counter() - t0, 3)

        result["processing_time"] = elapsed
        result["success"] = result.get("status", "ok") == "ok"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# WATER ANALYSIS
# ------------------------------------------------------------------------------

@app.route("/api/analyze-water", methods=["POST"])
def analyze_water():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = _read_image(request.files["image"])
        img_np = np.array(img)

        t0 = time.perf_counter()
        wq = analyze_water_quality(img_np)
        elapsed = round(time.perf_counter() - t0, 3)

        return jsonify({
            "success": True,
            **wq,
            "processing_time": elapsed,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# GALLERY
# ------------------------------------------------------------------------------

@app.route("/api/gallery")
def gallery_list():
    try:
        if os.path.exists(config.GALLERY_JSON):
            with open(config.GALLERY_JSON, "r") as f:
                entries = json.load(f)
        else:
            entries = []
        return jsonify({"success": True, "entries": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gallery/save", methods=["POST"])
def gallery_save():
    try:
        data = request.get_json(force=True)
        entry = {
            "enhanced_b64": data.get("enhanced_b64", ""),
            "filename":     data.get("filename", "image"),
            "saved_at":     time.strftime("%H:%M:%S"),
            "psnr":         data.get("psnr"),
            "ssim":         data.get("ssim"),
            "uiqm":         data.get("uiqm"),
            "uciqe":        data.get("uciqe"),
            "processing_time": data.get("processing_time"),
        }
        entries = []
        if os.path.exists(config.GALLERY_JSON):
            with open(config.GALLERY_JSON, "r") as f:
                entries = json.load(f)
        entries.insert(0, entry)          # newest first
        entries = entries[:100]           # cap at 100
        with open(config.GALLERY_JSON, "w") as f:
            json.dump(entries, f)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gallery/clear", methods=["DELETE"])
def gallery_clear():
    try:
        if os.path.exists(config.GALLERY_JSON):
            os.remove(config.GALLERY_JSON)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# STATUS
# ------------------------------------------------------------------------------

@app.route("/api/status")
def status():
    model = _load_model()
    yolo_ok = _load_detector() is not None
    midas_ok = _load_depth() is not None

    return jsonify({
        "model_loaded": model is not None,
        "device": str(config.DEVICE),
        "parameters": count_parameters(model) if model else None,
        "modules": {
            "yolo": yolo_ok,
            "midas": midas_ok,
            "opencv": True,
        },
    })


# ------------------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nJalDrishti FULL API Running...\n")
    _load_model()        # Pre-load U-Net enhancement model
    # Depth model loads lazily on first /api/depth request
    app.run(host="0.0.0.0", port=5500, debug=False, threaded=True)


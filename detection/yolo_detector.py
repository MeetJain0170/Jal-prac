"""
detection/yolo_detector.py — YOLO-based maritime object detection.

Uses Ultralytics YOLOv8. Install: pip install ultralytics

Maritime-relevant class mapping (COCO-trained base classes remapped):
    person          → Diver
    sports ball     → Naval Mine (spherical)
    boat/ship       → Surface Vessel
    Various objects → Security Threat or Marine Life
"""
from __future__ import annotations
import io
import base64
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Lazy import so the rest of the project works even without ultralytics
_ultralytics_available = False
_YOLO = None

def _get_yolo():
    global _ultralytics_available, _YOLO
    if _YOLO is not None:
        return _YOLO
    try:
        from ultralytics import YOLOWorld
        _YOLO = YOLOWorld
        _ultralytics_available = True
    except ImportError:
        try:
            from ultralytics import YOLO as _UltralyticsYOLO
            _YOLO = _UltralyticsYOLO
            _ultralytics_available = True
        except ImportError:
            pass
    return _YOLO


# ── Class remapping for maritime context ─────────────────────────────────────

MARITIME_REMAP: Dict[str, str] = {
    "diver":       "Diver",
    "naval mine":  "Naval Mine",
    "package":     "Suspicious Package",
    "hull damage": "Hull Damage",
    "weapon":      "Weapon",
    "submarine":   "Submersible",
    "boat":        "Surface Vessel",
    "ship":        "Surface Vessel",
    # Legacy COCO fallbacks
    "person":      "Diver",
    "sports ball": "Naval Mine",
    "car":         "Submersible",
    "backpack":    "Equipment",
    "umbrella":    "Equipment",
    "suitcase":    "Suspicious Package",
    "scissors":    "Sharp Debris",
    "knife":       "Weapon",
}

SECURITY_THREAT_CLASSES = {
    "Diver", "Naval Mine", "Submersible", "Weapon",
    "Suspicious Package", "Hull Damage",
}

MARINE_LIFE_CLASSES = {
    "Fish", "Shark", "Whale", "Dolphin",
    "Jellyfish", "Turtle", "Ray", "Marine Animal", "Coral",
}

ALERT_COLOURS = {
    "Security Threat": (255,  60,  60),
    "Marine Life":     ( 60, 210, 120),
    "Surface Vessel":  (255, 200,  50),
    "Object":          (100, 170, 255),
}


def _classify_category(label: str) -> str:
    remapped = MARITIME_REMAP.get(label, label.title())
    if remapped in SECURITY_THREAT_CLASSES:
        return "Security Threat"
    if remapped in MARINE_LIFE_CLASSES:
        return "Marine Life"
    return "Object"


# ── Detector class ────────────────────────────────────────────────────────────

class MaritimeDetector:
    """
    Wraps Ultralytics YOLO with maritime-specific post-processing.

    Parameters
    ----------
    weights     : model weight file or name ('yolov8n.pt', 'yolov8s.pt', …)
    conf_thresh : minimum detection confidence
    iou_thresh  : NMS IoU threshold
    img_size    : inference resolution
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh:  float = 0.45,
        img_size:    int   = 640,
    ):
        self.weights     = weights
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.img_size    = img_size
        self._model      = None

    def _load(self):
        YOLO = _get_yolo()
        if YOLO is None:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        if self._model is None:
            self._model = YOLO(self.weights)
            # Apply Open Vocabulary Prompts if using YOLOWorld
            if "world" in self.weights.lower() and hasattr(self._model, "set_classes"):
                self._model.set_classes([
                    "diver", "scuba diver", "person swimming", "human", "person",
                    "naval mine", "submarine", "submersible", "torpedo", 
                    "weapon", "kalashnikov", "rifle", "assault rifle", "gun", "knife",
                    "suspicious package", "package", "explosive", "hull damage",
                    "fish", "shark", "whale", "dolphin", "jellyfish", "turtle", "ray", "coral", "marine life",
                    "boat", "ship", "vessel", "shipwreck", 
                    "rock", "plant", "equipment"
                ])
        return self._model

    def detect(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single image.

        Parameters
        ----------
        img_np : HxWx3 uint8 numpy array

        Returns
        -------
        list of dicts:
            class, display_class, confidence, bbox [x1,y1,x2,y2], category
        """
        model = self._load()
        results = model(
            img_np,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                display = MARITIME_REMAP.get(label, label.title())
                cat     = _classify_category(label)
                detections.append({
                    "class":         label,
                    "display_class": display,
                    "confidence":    round(conf, 3),
                    "bbox":          [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    "category":      cat,
                })

        return detections

    def detect_and_annotate(
        self, img_np: np.ndarray
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Run detection and return (detections, annotated_image_base64).
        """
        detections = self.detect(img_np)
        annotated  = _draw_boxes(img_np.copy(), detections)
        img_pil    = Image.fromarray(annotated)
        buf        = io.BytesIO()
        img_pil.save(buf, format="PNG")
        b64        = base64.b64encode(buf.getvalue()).decode()
        return detections, f"data:image/png;base64,{b64}"


# ── Annotation drawing ────────────────────────────────────────────────────────

def _draw_boxes(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of img."""
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cat   = d.get("category", "Object")
        colour = ALERT_COLOURS.get(cat, (200, 200, 200))
        lbl   = f"{d['display_class']}  {d['confidence']:.0%}"

        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        text_y = max(0, y1 - 18)
        draw.rectangle([x1, text_y, x1 + len(lbl)*7, text_y+16], fill=colour)
        draw.text((x1+2, text_y), lbl, fill=(0, 0, 0))

    return np.array(pil_img)


# ── Module-level convenience ──────────────────────────────────────────────────

_default_detector: Optional[MaritimeDetector] = None


def get_detector(
    weights:     str   = "yolov8n.pt",
    conf_thresh: float = 0.35,
    iou_thresh:  float = 0.45,
    img_size:    int   = 640,
) -> MaritimeDetector:
    """Return (and cache) a singleton detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = MaritimeDetector(weights, conf_thresh, iou_thresh, img_size)
    return _default_detector
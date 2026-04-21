from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _draw(img_np: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    if img_np.ndim == 2:
        import cv2
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.shape[2] == 4:
        import cv2
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    im = Image.fromarray(img_np)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    colors = {
        "Marine Life": (52, 211, 153),
        "Diver": (251, 191, 36),
        "Security Threat": (239, 68, 68),
        "Surface Vessel": (96, 165, 250),
        "Object": (156, 163, 175),
    }

    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        c = colors.get(d.get("category", "Object"), (156, 163, 175))
        dr.rectangle([x1, y1, x2, y2], outline=c, width=2)
        label = f"{d.get('display_class', d.get('class', 'Object'))} {int(float(d.get('confidence', 0))*100)}%"
        dr.rectangle([x1, max(0, y1 - 16), x1 + len(label) * 7 + 6, y1], fill=c)
        dr.text((x1 + 3, max(0, y1 - 14)), label, fill=(0, 0, 0), font=font)

    return np.array(im)


class HybridDetector:
    """
    Combine:
    - custom marine model (fish/trash/etc. from the pipeline training)
    - pretrained diver model (COCO person → Diver) to keep legacy functionality
    """

    def __init__(
        self,
        marine_detector,
        diver_detector,
    ):
        self.marine_detector = marine_detector
        self.diver_detector = diver_detector

    def detect(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []

        try:
            dets.extend(self.marine_detector.detect(img_np))
        except Exception as e:
            logger.warning("Marine detector failed: %s", e)

        try:
            diver_dets = self.diver_detector.detect(img_np)
            diver_dets = [d for d in diver_dets if d.get("display_class") == "Diver" or d.get("category") == "Diver"]
            dets.extend(diver_dets)
        except Exception as e:
            logger.warning("Diver detector failed: %s", e)

        dets.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return dets

    def detect_and_annotate(
        self,
        inference_img: np.ndarray,
        original_img: Optional[np.ndarray] = None,
        context: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        dets = self.detect(inference_img)
        canvas = original_img if original_img is not None else inference_img
        ann = _draw(canvas, dets)
        pil = Image.fromarray(ann)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return dets, f"data:image/png;base64,{b64}"


_detector: Optional[HybridDetector] = None
_sig: Optional[tuple] = None


def get_detector(
    *,
    marine_weights: str,
    marine_conf: float,
    marine_iou: float,
    marine_imgsz: int,
    diver_weights: str,
    diver_conf: float,
    diver_iou: float,
    diver_imgsz: int,
) -> HybridDetector:
    global _detector, _sig
    sig = (marine_weights, marine_conf, marine_iou, marine_imgsz, diver_weights, diver_conf, diver_iou, diver_imgsz)
    if _detector is None or _sig != sig:
        from detection.simple_detector import get_detector as get_marine
        from detection.yolo_detector import get_detector as get_diver

        marine = get_marine(
            weights=marine_weights,
            conf_thresh=marine_conf,
            iou_thresh=marine_iou,
            img_size=marine_imgsz,
            tta=False,
            multi_scale=False,
        )

        diver = get_diver(
            weights=diver_weights,
            conf_thresh=diver_conf,
            iou_thresh=diver_iou,
            img_size=diver_imgsz,
            enhance=False,
            tta=False,
            run_world_pass=False,
        )

        _detector = HybridDetector(marine_detector=marine, diver_detector=diver)
        _sig = sig
    return _detector


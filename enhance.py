from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2

import config
from analysis.image_quality import compute_all_metrics
from depth.depth_estimator import estimate_depth_tensor

_to_tensor = transforms.ToTensor()


def apply_clahe_lab(img_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to the L channel of LAB colorspace for HDR-style local contrast enhancement."""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# PURE OPENCV MULTI-STAGE ENHANCEMENT (PHYSICS/OPTICS BASED)
# ─────────────────────────────────────────────────────────────────────────────

A_SHIFT = 0
B_SHIFT = 0
RED_STRENGTH = 30
CLAHE_CLIP = 1.2
OMEGA = 0.75
T_MIN = 0.35

def white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A = A - (np.mean(A) - 128) + A_SHIFT
    B = B - (np.mean(B) - 128) + B_SHIFT
    A = np.clip(A, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def restore_red(img):
    b, g, r = cv2.split(img)
    boost = cv2.equalizeHist(r)
    strength = RED_STRENGTH / 100.0
    r_new = cv2.addWeighted(r, 1 - strength, boost, strength, 0)
    return cv2.merge([b, g, r_new])

def clahe_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(CLAHE_CLIP, 0.1))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

def dehaze(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    A = np.percentile(gray, 95)
    t = 1 - OMEGA * (gray / A)
    t = np.clip(t, T_MIN, 1.0)
    t = cv2.merge([t, t, t])
    J = (img.astype(np.float32) - A) / t + A
    return np.clip(J, 0, 255).astype(np.uint8)

def sharpen(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.addWeighted(img, 1.2, blur, -0.2, 0)

def gamma_correct(img, g=1.1):
    inv = 1.0 / g
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def enhance_opencv(pil_img):
    """Pure classical optics pipeline (No AI). Returns PIL RGB."""
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    img = white_balance(img_bgr)
    img = restore_red(img)
    img = clahe_enhance(img)
    img = dehaze(img)
    img = sharpen(img)
    img = gamma_correct(img)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def enhance_image(model, pil_img, use_hybrid=True):
    """
    Hybrid Pipeline: If use_hybrid=True (default), it passes the image through the 
    6-stage OpenCV optics script FIRST to crush haze and restore reds, then passes 
    that optically-perfect image into the ResNet34 UNet for topological sharpening.
    """
    # 1. OPTIONAL: Optically pre-process raw image using the physics pipeline
    if use_hybrid:
        img = enhance_opencv(pil_img)
    else:
        img = pil_img.convert("RGB")

    orig_w, orig_h = img.size
    img_np = np.array(img)

    img_tensor = _to_tensor(img_np).unsqueeze(0).to(config.DEVICE)

    pad_h = (32 - orig_h % 32) % 32
    pad_w = (32 - orig_w % 32) % 32

    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        pred = model(img_tensor)

    alpha = float(getattr(config, "BLEND_ALPHA", 0.85))
    out_tensor = alpha * pred + (1 - alpha) * img_tensor

    out_tensor = out_tensor[0, :, :orig_h, :orig_w]
    out_tensor = torch.clamp(out_tensor, 0, 1)

    out_np = out_tensor.cpu().permute(1, 2, 0).numpy()
    out_np = (out_np * 255).astype(np.uint8)

    # UNet output denoising + slight CLAHE
    out_np = cv2.bilateralFilter(out_np, d=9, sigmaColor=75, sigmaSpace=75)
    out_np = apply_clahe_lab(out_np, clip_limit=1.5)

    return Image.fromarray(out_np)


def calculate_metrics_full(original, enhanced):

    orig_np = np.array(original.convert("RGB"), dtype=np.uint8)

    enh_np = np.array(enhanced.convert("RGB"), dtype=np.uint8)

    return compute_all_metrics(orig_np, enh_np)
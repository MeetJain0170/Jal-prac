# Introduction

## Background
Underwater vision is difficult because light absorption and scattering reduce color fidelity, contrast, and visibility. Standard camera pipelines are not sufficient for marine surveillance and ecological observation.

## Motivation
- Improve readability of underwater images for humans and models.
- Enable practical marine detection in noisy and hazy scenes.
- Build a usable system for academic demonstration and field-like examples.

## Existing System
Existing approaches usually focus on either enhancement or detection alone. Many systems perform poorly when applied to real underwater images with heavy color cast and suspended particles.

## Limitations of Existing Systems
- Weak performance in low-visibility scenes.
- Over-stylized enhancement artifacts.
- False detections due to underwater noise.
- Limited integration with user-friendly interfaces.

## Proposed Solution
JalDrishti provides an integrated pipeline with:
- Hybrid enhancement (deep model + classical OpenCV fallback).
- YOLO-based marine/diver detection with scene-aware post-processing.
- Depth estimation and environmental quality metrics.
- Web UI for upload, analyze, and visual reporting.

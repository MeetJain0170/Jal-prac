# Literature Survey

## Related Work Summary

### 1) U-Net for Low-Level Vision Restoration
U-Net style encoder-decoder architectures with skip connections remain a strong baseline for restoration tasks due to multi-scale feature fusion and spatial detail retention. This informed our enhancement backbone direction.

### 2) Classical Underwater Enhancement Methods
Methods like CLAHE, white-balance correction, and dehazing are computationally efficient and interpretable, but can over-amplify noise or introduce color instability when used alone.

### 3) YOLO-Based Real-Time Detection
YOLO models are widely adopted for practical detection because of good speed-accuracy balance. However, underwater domain shift often causes class confusion unless preprocessing and postprocessing are adapted.

### 4) Monocular Depth Estimation (MiDaS)
Depth estimation improves scene understanding beyond 2D labels and can help in interpretability for marine analysis use cases where depth sensors are absent.

### 5) Hybrid Vision Pipelines
Recent system-level practice favors combining learned models with rule-based guards for robustness. JalDrishti follows this principle to handle low-visibility edge cases.

## Existing Tools / Technologies Reviewed
- **OpenCV**: efficient image operators and visualization utilities
- **PyTorch**: deep-learning training/inference ecosystem
- **Ultralytics YOLO**: detection pipeline support
- **Flask**: API serving and browser integration
- **Pillow/NumPy**: core data manipulation

## Comparative Insight Table
| Method Class | Pros | Cons | Relevance to JalDrishti |
|---|---|---|---|
| Classical enhancement | Fast, deterministic, no heavy model dependency | Can overprocess colors/noise | Used as fallback and stabilization |
| Deep enhancement | Better restoration quality potential | Sensitive to checkpoint quality/domain | Used as primary hybrid path |
| Raw detection only | Simple pipeline | Unstable on underwater degradation | Avoided |
| Enhanced-image detection | Better separability and confidence | Needs careful relabel/cleanup | Adopted as core detection strategy |
| Hybrid postprocessing | Practical robustness | Added logic complexity | Adopted for final system quality |

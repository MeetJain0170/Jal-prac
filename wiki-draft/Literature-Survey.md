# Literature Survey

## Related Work Summary
1. **U-Net for image restoration tasks**  
   Encoder-decoder with skip connections is effective for pixel-level enhancement and has strong adoption in low-level vision.

2. **Underwater image enhancement methods (CLAHE, white-balance, dehazing)**  
   Classical approaches improve contrast quickly but can create over-processed artifacts if not balanced.

3. **YOLO family for real-time detection**  
   YOLO provides efficient object detection with practical performance for edge and demo scenarios.

4. **MiDaS depth estimation**  
   Monocular depth methods help infer scene structure where stereo or sonar is unavailable.

5. **Hybrid pipelines in adverse-visual domains**  
   Combining learned and heuristic modules improves robustness compared to single-model systems.

## Existing Tools / Technologies
- OpenCV for image processing
- PyTorch for deep-learning models
- Ultralytics YOLO for object detection
- Flask for serving model APIs

## Comparison Table
| Approach | Strength | Limitation |
|---|---|---|
| Classical enhancement only | Fast, lightweight | Unstable color realism |
| Deep enhancement only | Better visual reconstruction | Fallback needed on model failure |
| Detection only | Direct object output | Struggles on degraded underwater input |
| **Hybrid JalDrishti pipeline** | Robust end-to-end behavior | Higher integration complexity |

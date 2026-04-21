# Methodology

## Step-by-Step Working
1. User uploads an underwater image through browser UI.
2. Backend normalizes and validates image input.
3. Hybrid enhancement runs:
   - deep model prediction
   - classical OpenCV enhancement
   - tone harmonization and fallback checks
4. Detection runs on enhanced image:
   - marine detector
   - diver detector
   - optional auxiliary recall pass for sparse marine scenes
5. Scene-aware postprocessing:
   - overlap de-duplication
   - false-positive suppression
   - context-based relabeling
6. Depth and environmental analysis are computed.
7. Results are returned as annotated images + JSON metrics.

## Algorithms Used
- U-Net-based image enhancement
- OpenCV-based contrast/color correction
- YOLO-based object detection
- Non-max suppression and heuristic relabeling
- MiDaS monocular depth estimation

## Flowcharts / Diagrams
Add flowchart image:
`<ADD_FLOWCHART_IMAGE>`

## Data Flow Explanation
Input image data flows from frontend to backend API, then sequentially through enhancement and detection pipelines. Detection outputs are refined using scene-aware logic before analytics modules generate depth/threat/environment metadata. Final response returns both visualization assets and structured metrics to the frontend.

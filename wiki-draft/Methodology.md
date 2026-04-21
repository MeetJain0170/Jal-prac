# Methodology

## Step-by-Step Working
1. User uploads underwater image.
2. API performs hybrid enhancement.
3. Detection runs on enhanced image.
4. Scene-aware post-processing cleans labels.
5. Depth and water-quality metrics are computed.
6. Annotated outputs and metrics are rendered in UI.

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
Input image data flows from frontend to backend API, through enhancement and detection pipelines, then to post-processing and analytics modules, and finally back to frontend as base64 images + JSON metadata.

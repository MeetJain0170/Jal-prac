# System Architecture

## Architecture Diagram
Upload an architecture image in wiki and link here:
`<ADD_ARCHITECTURE_IMAGE>`

Example sections to include in the diagram:
- Input Image
- Enhancement Engine
- Detection Engine
- Post-processing Layer
- Depth and Water Analysis
- Web UI Output Panel

## Architecture Explanation
JalDrishti uses a layered architecture:
1. **Input Layer**: user uploads image through web UI.
2. **Enhancement Layer**: deep enhancement + OpenCV fallback/tone stabilization.
3. **Detection Layer**: marine and diver detection on enhanced image.
4. **Refinement Layer**: scene-aware postprocess, relabel guards, de-duplication.
5. **Analytics Layer**: depth estimation + water/threat analysis.
6. **Presentation Layer**: annotated image, metrics, and recommendation panels.

## Modules / Components
- **Frontend UI (`static/`)**: upload, preview, result rendering.
- **Flask API (`api.py`)**: orchestrates pipeline endpoints.
- **Enhancement module (`enhance.py`)**: hybrid restoration path.
- **Detection modules (`detection/`)**: marine + diver detectors.
- **Depth module (`depth/`)**: monocular depth estimation.
- **Analysis modules (`analysis/`)**: threat and water indicators.

## Design Highlights
- Detection runs on enhanced imagery to improve recall.
- Fallback logic prevents blank outputs on model/runtime failures.
- Post-processing enforces quality constraints for practical scene consistency.

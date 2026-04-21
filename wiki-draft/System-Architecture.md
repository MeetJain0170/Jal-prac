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
The system takes a user-uploaded image and processes it through enhancement and analysis modules. Detection runs on enhanced imagery, then scene-aware post-processing refines labels and suppresses noise. Results are sent to UI panels for annotation, metrics, and environmental intelligence.

## Modules / Components
- **Frontend UI (`static/`)**: upload, preview, result rendering.
- **Flask API (`api.py`)**: orchestrates pipeline endpoints.
- **Enhancement module (`enhance.py`)**: hybrid restoration path.
- **Detection modules (`detection/`)**: marine + diver detectors.
- **Depth module (`depth/`)**: monocular depth estimation.
- **Analysis modules (`analysis/`)**: threat and water indicators.

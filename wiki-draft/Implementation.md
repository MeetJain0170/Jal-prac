# Implementation

## Project Setup Steps
```bash
git clone https://github.com/MeetJain0170/Jal-prac.git
cd Jal-prac
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python api.py
```

Open in browser: `http://localhost:5500`

## Code Structure (Folder Explanation)
- `api.py`: primary Flask API and orchestration logic
- `detection/`: marine, diver, and hybrid detector modules
- `depth/`: depth-estimation loader/inference
- `analysis/`: threat and water-quality analysis
- `static/`: frontend UI (HTML/CSS/JS)
- `outputs/checkpoints/`: model checkpoints (LFS)
- `data/JalDrishti/`: trained weights and artifacts (LFS)

## Key Code Snippets
```python
# Example endpoint flow
@app.route("/api/detect", methods=["POST"])
def detect():
    infer_img = _read_image(request.files["image"])
    infer_np = np.array(infer_img)
    detections, annotated = detector.detect_and_annotate(infer_np, original_img=infer_np)
    return jsonify({"success": True, "detections": detections, "annotated_image": annotated})
```

```python
# Example enhancement fallback idea
if model is None:
    enhanced_hybrid = enhance_opencv(img)
else:
    enhanced_hybrid = enhance_image(model, img, use_hybrid=True)
```

## Integration Details
- Enhancement output is used as detection input.
- Post-processing merges detector outputs and cleans labels.
- UI displays enhanced image, detection overlays, depth map, and analytics.
- LFS-managed model files are pulled during setup for reproducible behavior.

## Repository Link
[Jal-prac Repository](https://github.com/MeetJain0170/Jal-prac)

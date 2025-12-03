# Project Structure

Overview of key files and directories.

```
anemianew/
├─ main.py                    # FastAPI backend (inference API on port 8081)
├─ gradioApp.py               # Gradio frontend UI
├─ requirements.txt           # Python dependencies
├─ models/
│  ├─ eye_seg_model.pt        # YOLO segmentation weights
│  ├─ Hemoglobin_predictor.h5 # ANN regressor (Keras/TensorFlow)
│  ├─ input_scaler.pkl        # Scaler for RGB percentages (input)
│  └─ output_scaler.pkl       # Scaler for hemoglobin (output)
├─ user_history/              # Per-user flagged results as JSONL (created at runtime)
├─ test_images/               # Sample images for testing
├─ presentation_figs/         # Plots/figures for presentation
├─ anemia_ANN.ipynb           # Notebook (model analysis/experiments)
├─ train_anemia_yolov8.ipynb  # Notebook (YOLO training)
└─ README.md                  # Project overview and quickstart
```

## Data flow
1. Gradio app sends an image to FastAPI `/predict` as `multipart/form-data` (`file` field).
2. Backend runs YOLO segmentation, computes RGB percentages, scales inputs, and runs ANN.
3. Predicted hemoglobin is inverse-scaled and mapped to status (Anemic/Non-Anemic).
4. Frontend displays result; optional save to `user_history/USERNAME.jsonl`.

## Important functions
- `main.py`:
  - `load_models()` loads YOLO and ANN on startup.
  - `predict()` handles image upload and returns JSON response.
- `gradioApp.py`:
  - `call_api()` posts the image to the backend.
  - `process_image_from_upload()` / `process_image_from_camera()` pipelines UI to API.
  - `generate_diet_plan()` creates basic diet guidance based on Hb level.

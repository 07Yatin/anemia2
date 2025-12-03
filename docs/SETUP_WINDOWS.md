# Setup on Windows (PowerShell)

This guide walks you through setting up the project on a fresh Windows machine using PowerShell.

## Prerequisites
- Python 3.10 or 3.11 installed and on PATH
- Git installed and on PATH
- Internet access to install Python wheels (Torch, TensorFlow, etc.)

## 1) Clone the repository
```powershell
git clone https://github.com/07Yatin/anemianew.git
cd anemianew
```

## 2) Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If activation is blocked by execution policy:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `ultralytics` pulls a matching CPU PyTorch wheel automatically.
- If TensorFlow installation fails on your system, try the CPU-only wheel:
```powershell
pip uninstall -y tensorflow
pip install tensorflow-cpu==2.17.0
```

## 4) Verify model files
Ensure the following files exist in `models/`:
- `models/eye_seg_model.pt`
- `models/Hemoglobin_predictor.h5`
- `models/input_scaler.pkl`
- `models/output_scaler.pkl`

If any are missing, copy them into the `models/` folder before continuing.

## 5) Start the FastAPI backend
```powershell
python main.py
```
The backend runs at `http://127.0.0.1:8081`.

Quick API test in a second PowerShell window (replace image path as needed):
```powershell
$FilePath = ".\test_images\20251006_221810.jpg"
Invoke-RestMethod -Uri "http://127.0.0.1:8081/predict" -Method Post -Form @{ file = Get-Item $FilePath }
```

## 6) Start the Gradio frontend
Open a new PowerShell window in the project folder:
```powershell
.\.venv\Scripts\Activate.ps1
python gradioApp.py
```

The terminal prints a local URL (and optionally a public share URL). Open it in your browser.

## 7) Basic usage
- Use “Upload Image” or “Take Photo” to submit a conjunctiva image
- View hemoglobin estimate, anemia status, and diet tips
- Optionally “Flag / Save” a result for a username to store in `user_history/USERNAME.jsonl`

## Next steps
- See `docs/RUN_GUIDE.md` for API details and configuration
- See `docs/TROUBLESHOOTING.md` for common fixes

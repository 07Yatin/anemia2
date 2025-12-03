# Troubleshooting

Common issues and quick fixes when running locally on Windows.

## Dependency installation errors

- Upgrade pip first:
```powershell
pip install --upgrade pip
```

- TensorFlow fails to install or import:
```powershell
pip uninstall -y tensorflow
pip install tensorflow-cpu==2.17.0
```

- PyTorch/Ultralytics wheel issues: retry after pip upgrade; most CPU wheels resolve automatically.

## OpenCV import/runtime errors

- Install Microsoft Visual C++ Redistributable (x64) from Microsoft if missing.
- Ensure you are using the project virtual environment.

## Port already in use

- Change port in `main.py` (e.g., `port=8082`).
- Update `gradioApp.py` `call_api` URL to match the new port.

## API returns error / no mask detected

- Ensure the backend is running before starting the Gradio app.
- Use clear, well-lit images of the everted lower eyelid conjunctiva.
- If result shows default fallback, retake image in better lighting.

## Module not found

- Activate venv and reinstall requirements:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Slow first run

- Initial model loads and first inference may take longer due to graph/weights initialization.

## Still stuck?

- Share the exact error text and your Python version. Include the output of:
```powershell
python --version
pip list
```

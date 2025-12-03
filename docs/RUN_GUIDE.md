# Run Guide

This document covers how to run the backend and frontend, configure ports, and use the API directly.

## Start services

Backend (FastAPI):
```powershell
python main.py
```
Runs at `http://127.0.0.1:8081` by default.

Frontend (Gradio) in a second terminal:
```powershell
python gradioApp.py
```

## Configuration

- Backend port: adjust in `main.py` where `uvicorn.run(app, host="0.0.0.0", port=8081)` is called.
- Frontend API URL: `gradioApp.py` function `call_api` uses `http://127.0.0.1:8081/predict`. Update it if you change the backend host/port.

## API usage (direct)

Endpoint: `POST /predict`

Content type: `multipart/form-data` with field name `file`

Example with Windows PowerShell:
```powershell
$FilePath = ".\test_images\20251006_221810.jpg"
Invoke-RestMethod -Uri "http://127.0.0.1:8081/predict" -Method Post -Form @{ file = Get-Item $FilePath }
```

Example with curl:
```bash
curl -X POST http://127.0.0.1:8081/predict -F "file=@test_images/20251006_221810.jpg"
```

Sample response:
```json
{
  "hgl": "11.52g/dl",
  "status": "Non-Anemic"
}
```

## Data & history

- User-flagged results are stored as JSON Lines files in `user_history/USERNAME.jsonl`.
- No images are persisted by default; only summary results are saved when you flag them.

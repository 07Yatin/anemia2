# Authentication Setup (MERN + FastAPI)

This project now includes a MERN-style authentication layer (MongoDB + Express/Node + React) and secures the FastAPI ML endpoint with JWT.

## Components
- `auth-service/`: Node/Express service with MongoDB, JWT auth (register/login/me)
- `web-frontend/`: React app (Vite) with Login/Register and image upload to ML service
- `main.py`: FastAPI ML API, now verifying JWTs in the `Authorization: Bearer <token>` header

## Prerequisites
- Node.js 18+
- MongoDB running locally (or a connection string to a remote cluster)
- Python 3.10/3.11 and the project venv

## 1) Start MongoDB
- Local default: `mongodb://localhost:27017`
- Or set `MONGO_URI` in `auth-service/.env`

## 2) Start Auth Service
```
cd auth-service
cp .env.example .env
# edit .env: set MONGO_URI and a strong JWT_SECRET
npm install
npm run dev
```
Runs on `http://localhost:4000`.

## 3) Start ML Backend (FastAPI)
From project root with venv active:
```
pip install -r requirements.txt
$env:JWT_SECRET = "<same-secret-as-auth-service>"
$env:AUTH_ENABLED = "1"
python main.py
```
Runs on `http://127.0.0.1:8081`.

## 4) Start React Frontend
```
cd web-frontend
npm install
# optional: create .env with VITE_AUTH_BASE and VITE_ML_BASE
npm run dev
```
Open the printed URL (default `http://localhost:5173`).

## 5) Flow
1. Register or Login in the React app → receives JWT
2. React stores token and uses it to call FastAPI `/predict`
3. FastAPI verifies token using `JWT_SECRET` and returns results

## Notes
- CORS is enabled in `main.py` for localhost dev ports
- Toggle auth via `AUTH_ENABLED=0` if needed (dev only)

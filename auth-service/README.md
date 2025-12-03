# Auth Service (Node/Express)

JWT-based authentication service for the Anemia app.

## Setup

1) Copy .env:
```
cp .env.example .env
```
2) Edit `.env` with your Mongo URI and JWT secret.

3) Install and run:
```
npm install
npm run dev
```
Service listens on `PORT` (default 4000).

## Endpoints
- POST /api/auth/register { name, email, password }
- POST /api/auth/login { email, password }
- GET /api/auth/verify (Authorization: Bearer <token>)
- GET /api/auth/me (Authorization: Bearer <token>)

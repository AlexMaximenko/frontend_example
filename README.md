# WanderList

WanderList is a small demo project showcasing a modern web stack for exploring travel destinations. It includes a React + TypeScript frontend and a FastAPI backend with Docker-based development setup.

## Prerequisites

- Docker & Docker Compose
- Node.js 20+
- Python 3.11+

## Quick start (Docker Compose)

```bash
make up
# Frontend: http://localhost:5173
# Backend docs: http://localhost:8000/docs
```

Stop the stack with:

```bash
make down
```

## Manual run

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

## Useful commands

- `npm run lint` – run ESLint
- `npm run format` – run Prettier
- `uvicorn app.main:app --reload` – run backend with reload

## Project structure

```
wanderlist/
  backend/
    app/
      routers/
      static/images/
      ...
  frontend/
    src/
      routes/
      components/
      lib/
      styles/
```

## Notes

- CORS is configured for `http://localhost:5173`.
- Sample images live under `backend/app/static/images`.
- To add destinations, edit `backend/app/data.py`.

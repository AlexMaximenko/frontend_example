.PHONY: up down fmt lint

up:
docker compose up --build

down:
docker compose down

fmt:
cd backend && ruff format .
cd frontend && npm run format

lint:
cd backend && ruff check .
cd frontend && npm run lint

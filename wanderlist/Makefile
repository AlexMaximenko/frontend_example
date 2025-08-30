.PHONY: up down fmt lint

up:
	docker compose up --build

down:
	docker compose down

fmt:
	cd frontend && npm run format
	cd backend && python -m ruff format .

lint:
	cd frontend && npm run lint
	cd backend && python -m ruff check .
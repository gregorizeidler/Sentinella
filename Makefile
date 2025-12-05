.PHONY: help install test run docker-build docker-up docker-down clean

help:
	@echo "Sentinella AI Gateway - Makefile Commands"
	@echo ""
	@echo "  make install      - Install Python dependencies"
	@echo "  make test         - Run tests"
	@echo "  make run          - Run gateway locally"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start services with Docker Compose"
	@echo "  make docker-down  - Stop Docker Compose services"
	@echo "  make clean        - Clean Python cache files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	uvicorn src.gateway.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sentinella-gateway:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete


# Makefile for EduGrade AI
.PHONY: help install install-dev test test-unit test-integration test-coverage lint format clean run-api run-dashboard-streamlit run-dashboard-gradio run-docker build-docker stop-docker logs

# Default target
help:
	@echo "EduGrade AI - Multi-Agentic Answer Sheet Evaluator"
	@echo ""
	@echo "Available targets:"
	@echo "  install              Install production dependencies"
	@echo "  install-dev          Install development dependencies"
	@echo "  test                 Run all tests"
	@echo "  test-unit            Run unit tests only"
	@echo "  test-integration     Run integration tests only"
	@echo "  test-coverage        Run tests with coverage report"
	@echo "  lint                 Run code linting"
	@echo "  format               Format code with black"
	@echo "  clean                Clean temporary files"
	@echo "  run-api              Run FastAPI backend"
	@echo "  run-dashboard-streamlit  Run Streamlit dashboard"
	@echo "  run-dashboard-gradio Run Gradio dashboard"
	@echo "  run-docker           Run with Docker Compose"
	@echo "  build-docker         Build Docker images"
	@echo "  stop-docker          Stop Docker containers"
	@echo "  logs                 Show Docker logs"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Testing
test:
	python tests/run_tests.py --all --verbose

test-unit:
	python tests/run_tests.py --unit --verbose

test-integration:
	python tests/run_tests.py --integration --verbose

test-coverage:
	python tests/run_tests.py --unit --coverage --verbose

# Code quality
lint:
	python tests/run_tests.py --lint

format:
	python tests/run_tests.py --format

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf uploads/*
	rm -rf processed/*
	rm -rf exports/*
	rm -f *.db
	rm -f *.db-journal

# Running applications
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard-streamlit:
	streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

run-dashboard-gradio:
	python dashboard/gradio_app.py

# Docker operations
run-docker:
	docker-compose up --build

build-docker:
	docker-compose build

stop-docker:
	docker-compose down

logs:
	docker-compose logs -f

# Development setup
setup-dev: install-dev
	@echo "Setting up development environment..."
	mkdir -p uploads processed grades exports models
	@echo "Development environment ready!"
	@echo "Run 'make setup-gemini' to configure Gemini API"
	@echo "Run 'make run-api' to start the backend"
	@echo "Run 'make run-dashboard-streamlit' to start Streamlit dashboard"
	@echo "Run 'make run-dashboard-gradio' to start Gradio dashboard"

# Gemini API setup
setup-gemini:
	python scripts/setup_gemini.py

# Production setup
setup-prod: install
	@echo "Setting up production environment..."
	mkdir -p uploads processed grades exports models
	@echo "Production environment ready!"
	@echo "Run 'make run-docker' to start with Docker"

# Quick start
quick-start: setup-dev
	@echo "Starting EduGrade AI in development mode..."
	@echo "Backend will be available at: http://localhost:8000"
	@echo "Streamlit dashboard will be available at: http://localhost:8501"
	@echo "Gradio dashboard will be available at: http://localhost:7860"
	@echo ""
	@echo "Press Ctrl+C to stop all services"
	@echo ""
	@echo "Starting services..."
	@echo "Starting API..."
	@(make run-api &) && sleep 5
	@echo "Starting Streamlit dashboard..."
	@(make run-dashboard-streamlit &) && sleep 3
	@echo "Starting Gradio dashboard..."
	@(make run-dashboard-gradio &)
	@echo "All services started!"
	@echo "Visit the dashboards to get started."

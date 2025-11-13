.PHONY: help install test lint typecheck format security ci docker-build docker-test docker-ci clean

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Atsiitsʼiin - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install dependencies in virtual environment
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -e ".[dev]"

test: ## Run tests with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	pytest -v --cov=atsiitsiin --cov-report=term-missing

test-fast: ## Run tests without coverage
	@echo "$(BLUE)Running tests (fast mode)...$(NC)"
	pytest -v

test-html: ## Run tests with HTML coverage report
	@echo "$(BLUE)Running tests with HTML coverage...$(NC)"
	pytest -v --cov=atsiitsiin --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated: htmlcov/index.html$(NC)"
	@echo "$(YELLOW)Opening report in browser...$(NC)"
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Open htmlcov/index.html in your browser"

coverage-badge: ## Generate coverage badge SVG
	@echo "$(BLUE)Generating coverage badge...$(NC)"
	@coverage-badge -o coverage.svg -f 2>/dev/null || echo "Install coverage-badge: pip install coverage-badge"
	@echo "$(GREEN)Badge generated: coverage.svg$(NC)"

lint: ## Run ruff linter
	@echo "$(BLUE)Running ruff...$(NC)"
	ruff check .

lint-fix: ## Run ruff with auto-fix
	@echo "$(BLUE)Running ruff with auto-fix...$(NC)"
	ruff check . --fix

typecheck: ## Run pyright type checker
	@echo "$(BLUE)Running pyright...$(NC)"
	pyright

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format .
	black .

security: ## Run security scan with bandit
	@echo "$(BLUE)Running security scan...$(NC)"
	bandit -r src/atsiitsiin -c pyproject.toml --exclude "**/tests/*,**/test_*.py"

ci: lint typecheck security test ## Run all CI checks locally
	@echo "$(GREEN)All CI checks passed!$(NC)"

# Docker commands
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t atsiitsiin:latest --target development .

docker-test: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	docker-compose run --rm test

docker-test-html: ## Run tests in Docker with HTML coverage report
	@echo "$(BLUE)Running tests in Docker with HTML coverage...$(NC)"
	docker-compose run --rm test-html
	@echo "$(GREEN)Coverage report generated: htmlcov/index.html$(NC)"
	@echo "$(YELLOW)Opening report in browser...$(NC)"
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Open htmlcov/index.html in your browser"

docker-lint: ## Run linting in Docker
	@echo "$(BLUE)Running linters in Docker...$(NC)"
	docker-compose run --rm lint

docker-ci: ## Run all CI checks in Docker
	@echo "$(BLUE)Running CI pipeline in Docker...$(NC)"
	docker-compose run --rm ci

docker-shell: ## Open a shell in Docker container
	@echo "$(BLUE)Opening Docker shell...$(NC)"
	docker-compose run --rm dev bash

docker-clean: ## Remove Docker containers and images
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose down -v
	docker rmi atsiitsiin:latest || true

run-cli: ## Launch the Atsiitsʼiin CLI demo
	uv run python examples/run_atsiitsiin.py

run-api: ## Start the FastAPI backend with hot reload
	uv run uvicorn backend.app.main:app --reload

run-streamlit: ## Launch the Streamlit prototype UI
	uv run streamlit run frontend/streamlit_app.py

run-pipeline: ## Execute Snowpark sentiment pipeline
	uv run python pipelines/sentiment.py

# Cleanup commands
clean: ## Remove cache and build artifacts
	@echo "$(BLUE)Cleaning cache and build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/

clean-all: clean docker-clean ## Remove all artifacts including Docker
	@echo "$(GREEN)All cleaned!$(NC)"

# Pre-commit setup
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit...$(NC)"
	pre-commit run --all-files


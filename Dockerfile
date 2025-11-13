# Multi-stage Dockerfile for Atsiitsʼiin
# Stage 1: Base image with Python 3.13
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Stage 2: Dependencies
FROM base AS dependencies

# Install Node.js for pyright (required dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (for better caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Stage 3: Development (includes source code)
FROM dependencies AS development

# Copy the entire project
COPY . .

# Install the package in editable mode
RUN pip install -e ".[dev]"

# Default command: run all checks
CMD ["sh", "-c", "ruff check . && pyright && pytest"]

# Stage 4: Testing (optimized for CI)
FROM dependencies AS testing

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# Install package
RUN pip install -e .

# Default: run tests only
CMD ["pytest", "-v"]

# Stage 5: Linting (for running checks only)
FROM dependencies AS linting

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# Install package
RUN pip install -e .

# Run linters
CMD ["sh", "-c", "ruff check . && pyright"]


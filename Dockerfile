# Multi-stage Dockerfile for skdr-eval development and production

# Development stage
FROM python:3.11-slim as development

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install package in development mode
RUN pip install --no-cache-dir -e .[dev,examples]

# Copy source code
COPY . .

# Set up git (for development)
RUN git config --global --add safe.directory /workspace

# Default command for development
CMD ["bash"]

# Production stage
FROM python:3.11-slim as production

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy and install package
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
USER app

# Default command for production
CMD ["python", "-c", "import skdr_eval; print('skdr-eval is ready!')"]

FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies (build-essential for C-extensions like numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install Poetry \
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir poetry==2.1.3

# Copy dependencies first (for Docker layer caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies globally in the container
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the remaining application code \
COPY . .

# Ensure the internal output directory exists
RUN mkdir -p /app/data

# Default command (change cli arguments according to your preferences)
CMD ["python", "src/cli.py", "--conversations", "1", "--mode", "fill"]
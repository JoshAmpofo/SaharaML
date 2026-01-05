# Base image
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# set environment variables
ENV PATH="/app/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" ./

# Install uv (fast Python package mananger
RUN uv sync --locked

# Copy application code and artifacts
COPY src /app/src
COPY artifacts /app/artifacts
COPY data /app/data
COPY models /app/models

# Expose API port
EXPOSE 9696

# Run FastAPI app
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "9696"]
# Multi-stage build for longtext-pipeline
# Stage 1: Builder - install dependencies and build package
FROM python:3.10-slim AS builder

WORKDIR /build

# Copy pyproject.toml and install dependencies first (better caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -e ".[dev]"

# Stage 2: Runtime - minimal image for running the CLI
FROM python:3.10-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Ensure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy source code
COPY src/ longtext_pipeline/

# Copy entry point script
COPY --from=builder /build/pyproject.toml .

# Create output directories with proper permissions
RUN mkdir -p /data /output && chown -R appuser:appuser /app

USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Entry point is the longtext CLI (registered in pyproject.toml)
ENTRYPOINT ["longtext"]
CMD ["--help"]

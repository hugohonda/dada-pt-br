FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Set environment to force CPU-only PyTorch installation
ENV TORCH_CUDA_ARCH_LIST=""
ENV FORCE_CUDA="0"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install CPU-only PyTorch first to avoid CUDA dependencies
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install dependencies and project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment and source code from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/README.md /app/README.md

# Create necessary directories with proper permissions
RUN mkdir -p output logs datasets/raw final-data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set the default command
ENTRYPOINT ["/app/.venv/bin/python", "-m", "dadaptbr.main"]
CMD ["--help"]

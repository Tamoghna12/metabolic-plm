# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="metabolic-plm contributors"
LABEL description="Reproducible CUDA 12.4 + PyTorch (bf16) + ESM-2 (PEFT/LoRA) + COBRApy + MMseqs2 container for 12GB RTX 4070"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface

# Install system dependencies, GLPK solver for COBRApy, and MMseqs2 static binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    glpk-utils \
    libglpk-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2 AVX2 static binary for homology-aware clustering
RUN curl -fsSL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mmseqs.tar.gz \
    && tar -xzf /tmp/mmseqs.tar.gz -C /opt \
    && ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
    && rm /tmp/mmseqs.tar.gz \
    && mmseqs version

# Create non-root user for secure container execution
RUN useradd -m -u 1000 -s /bin/bash plmuser
WORKDIR /workspace

# Install uv for fast, deterministic dependency installation
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /usr/local/bin/

# Copy project configuration and source
COPY pyproject.toml README.md /workspace/
COPY configs/ /workspace/configs/
COPY src/ /workspace/src/
COPY tests/ /workspace/tests/

# Install Python package and dependencies into system environment
RUN uv pip install --system --no-cache ".[dev]"

RUN chown -R plmuser:plmuser /workspace
USER plmuser

CMD ["pytest", "-v"]

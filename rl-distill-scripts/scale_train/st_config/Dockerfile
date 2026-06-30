# syntax=docker/dockerfile:1.3
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV UV_INDEX_URL=https://pypi.org/simple
ENV HF_HOME=/tmp/hf_cache
ENV XDG_CACHE_HOME=/tmp/.cache
ENV DD_SERVICE=rl-distill

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    iproute2 \
    libnuma-dev \
    openssh-client \
    procps \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rl-distill

COPY . /workspace/rl-distill

RUN PIP_INDEX_URL=https://pypi.org/simple \
    UV_INDEX_URL=https://pypi.org/simple \
    bash rl-distill-scripts/setup_env.sh

ENV PATH="/workspace/rl-distill/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace/rl-distill:${PYTHONPATH}"

RUN python3 -c "import torch, vllm, flash_attn, transformers; print(torch.__version__, vllm.__version__, flash_attn.__version__, transformers.__version__)"

# The same source checkout powers CPU generation and GPU verification images.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu22.04

FROM python:3.10-slim AS agent-remote
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl jq ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app/KernelBench
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN curl -fsSL https://claude.ai/install.sh | bash
COPY . .
ENV PYTHONPATH=/app/KernelBench PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH
CMD ["/bin/bash"]

FROM ${CUDA_IMAGE} AS agent
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip build-essential git curl jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.10 /usr/local/bin/python
WORKDIR /app/KernelBench
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN curl -fsSL https://claude.ai/install.sh | bash
COPY . .
ENV PYTHONPATH=/app/KernelBench PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
CMD ["/bin/bash"]

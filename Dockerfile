FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common ca-certificates \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get update && apt-get install -y \
    curl wget git \
    python3 python3-pip python3-venv \
    gcc-13 g++-13 cmake build-essential \
    vim less jq \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
RUN pip3 install uv

# 3. Install Claude CLI
RUN curl -fsSL https://claude.ai/install.sh | bash
#  Add Claude CLI to PATH
ENV PATH="/root/.local/bin:/root/.anthropic/bin:$PATH"

# 4. Clone repository
WORKDIR /app
RUN git clone https://github.com/shadowfall09/KernelBench.git

# 5. Configure environment
WORKDIR /app/KernelBench

# Set C++ compiler environment variables (required by z3-solver)
ENV CC=gcc-13
ENV CXX=g++-13
ENV CXXFLAGS="-std=c++20"

RUN uv venv
RUN . .venv/bin/activate && uv sync --extra gpu

# 6. Prepare directories
RUN mkdir -p runs/claude_code cache

# 7. Set environment variables
ENV PATH="/app/KernelBench/.venv/bin:$PATH"
ENV PYTHONPATH="/app/KernelBench/src:$PYTHONPATH"

# Claude/Bedrock configuration (for generation tasks)
ENV CLAUDE_CODE_USE_BEDROCK=1
ENV ANTHROPIC_MODEL='global.anthropic.claude-opus-4-6-v1'
ENV ANTHROPIC_SMALL_FAST_MODEL='us.anthropic.claude-haiku-4-5-20251001-v1:0'

# Copy entrypoint script (used for generation)
COPY entrypoint.sh /app/KernelBench/entrypoint.sh
RUN chmod +x /app/KernelBench/entrypoint.sh

# Default working directory
WORKDIR /app/KernelBench

# No ENTRYPOINT - allows flexible use for both generation and evaluation
# For generation: docker run ... /app/KernelBench/entrypoint.sh
# For evaluation: docker run ... bash -c "uv run python scripts/eval_from_generations.py ..."
CMD ["/bin/bash"]
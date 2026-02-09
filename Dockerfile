FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install dependencies
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

# 设置 C++ 编译器环境变量 (z3-solver 需要)
ENV CC=gcc-13
ENV CXX=g++-13
ENV CXXFLAGS="-std=c++20"

RUN uv venv
RUN . .venv/bin/activate && uv sync --extra gpu

# 6. Prepare directories and scripts
RUN mkdir -p runs/claude_code
COPY entrypoint.sh /app/KernelBench/entrypoint.sh
RUN chmod +x /app/KernelBench/entrypoint.sh

ENV PATH="/app/KernelBench/.venv/bin:$PATH"
ENV CLAUDE_CODE_USE_BEDROCK=1

ENTRYPOINT ["/app/KernelBench/entrypoint.sh"]
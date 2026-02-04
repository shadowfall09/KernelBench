FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装基础依赖
RUN apt-get update && apt-get install -y \
    curl git python3 python3-pip python3-venv vim less \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 uv
RUN pip3 install uv

# 3. 安装 Claude Code (官方脚本)
RUN curl -fsSL https://claude.ai/install.sh | bash
# 将安装路径加入 PATH
ENV PATH="/root/.local/bin:/root/.anthropic/bin:$PATH"

# 4. 克隆仓库
WORKDIR /app
RUN git clone https://github.com/shadowfall09/KernelBench.git

# 5. 配置环境
WORKDIR /app/KernelBench
RUN uv venv
RUN . .venv/bin/activate && uv sync --extra gpu

# 6. 准备目录和脚本
RUN mkdir -p runs/claude_code
COPY entrypoint.sh /app/KernelBench/entrypoint.sh
RUN chmod +x /app/KernelBench/entrypoint.sh

ENV PATH="/app/KernelBench/.venv/bin:$PATH"
ENV CLAUDE_CODE_USE_BEDROCK=1

ENTRYPOINT ["/app/KernelBench/entrypoint.sh"]
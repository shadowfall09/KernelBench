docker build -t kb-claude:v1 .

# 确保本地有输出目录
mkdir -p $(pwd)/runs_output

# 运行 Level 1, Problem 5
docker run --rm -it \
    --gpus '"device=2"' \
    -e AWS_BEARER_TOKEN_BEDROCK="token" \
    -e KB_LEVEL=1 \
    -e KB_PROBLEM=5 \
    -v $(pwd)/runs_output:/app/KernelBench/runs/claude_code \
    kb-claude:v1
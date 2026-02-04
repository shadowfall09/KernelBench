docker build -t kb-claude-dynamic:v1 .

# 确保本地有输出目录
mkdir -p $(pwd)/runs_output

# 运行 Level 1, Problem 5
docker run --rm -it \
    --gpus all \
    -e AWS_ACCESS_KEY_ID="你的AK" \
    -e AWS_SECRET_ACCESS_KEY="你的SK" \
    -e AWS_SESSION_TOKEN="你的TOKEN" \
    -e KB_LEVEL=1 \
    -e KB_PROBLEM=5 \
    -v $(pwd)/runs_output:/app/KernelBench/runs/claude_code \
    kb-claude-dynamic:v1
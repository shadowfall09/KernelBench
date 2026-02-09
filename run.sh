# docker build -t kb-claude:v1 .

LEVEL=${1:-1}
PROBLEM=${2:-6}
DEVICE=${3:-2}

mkdir -p runs_output
mkdir -p $(pwd)/runs_output/${LEVEL}_${PROBLEM}

docker run --rm -it \
    --gpus "\"device=${DEVICE}\"" \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    -e AWS_BEARER_TOKEN_BEDROCK="${AWS_BEARER_TOKEN_BEDROCK:-}" \
    -e KB_LEVEL=$LEVEL \
    -e KB_PROBLEM=$PROBLEM \
    -v $(pwd)/runs_output/${LEVEL}_${PROBLEM}:/app/KernelBench/runs/claude_code \
    kb-claude:v1
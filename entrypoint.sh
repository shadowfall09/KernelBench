#!/bin/bash
set -e

# --- 1. 检查环境变量 ---

# 检查 AWS 凭证
# if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_SESSION_TOKEN" ]; then
#     echo "❌ Error: AWS credentials are missing!"
#     exit 1
# fi

# 检查任务参数 (默认为 Level 1, Problem 5)
KB_LEVEL=${KB_LEVEL:-1}
KB_PROBLEM=${KB_PROBLEM:-5}

echo "=========================================================="
echo "🚀 Starting Claude Code Agent (AWS Bedrock Mode)"
echo "🎯 Target: Level $KB_LEVEL, Problem $KB_PROBLEM"
echo "📂 Output: runs/claude_code"
echo "========================================================"

# --- 2. 构建动态 Prompt ---

AGENT_PROMPT=$(cat <<EOF
You are an expert CUDA engineer, specialized in writing high-performance GPU kernels on NVIDIA RTX A6000 (Ampere architecture).
Your task is to solve **Level $KB_LEVEL, Problem $KB_PROBLEM** in the KernelBench repository located in the current directory.
You must write a CUDA kernel that is both correct and optimized for performance. If you are unable to optimize further in 5 rounds, provide a correct implementation.

Rules:
1. Do NOT run scripts/generate_samples.py.
2. You may read its logic to understand the required output format and conventions.

Implementation:
- Reference implementations are in: KernelBench/KernelBench
- Write your solution under: KernelBench/runs/claude_code
- The 'example' folder under 'runs' already exists and contains an example kernel with no speedup.
- Preserve the file naming pattern in 'claude_code'.
- Write a correct and optimized CUDA implementation compatible with KernelBench.

Evaluation Command:
You MUST run the following command to verify your solution. 
Note: The parameter \`subset="($KB_PROBLEM,$KB_PROBLEM)"\` explicitly tells the script to ONLY test Problem $KB_PROBLEM.

\`\`\`bash
uv run python scripts/eval_from_generations.py \\
  run_name=claude_code \\
  dataset_src=local \\
  level=$KB_LEVEL \\
  num_gpu_devices=1 \\
  timeout=300 \\
  subset="($KB_PROBLEM,$KB_PROBLEM)" \\
  gpu_arch="['Ampere']"
\`\`\`

Results will be written to: KernelBench/runs/claude_code/eval_results.json
Delete this file first if you need to evaluate again.

Optimization Goal:
Iterate on the kernel until performance is maximized while maintaining correctness.
EOF
)

# --- 3. 执行 Claude Code ---

echo ""
echo ">>> Sending prompt to Claude Code..."

timeout 1800 claude -p "$AGENT_PROMPT" --allowedTools "Read,Edit,Bash" --output-format stream-json --verbose --include-partial-messages


# uv run python scripts/eval_from_generations.py \
#   run_name=example \
#   dataset_src=local \
#   level=1 \
#   num_gpu_devices=1 \
#   timeout=300 \
#   subset="(1,1)" \
#   gpu_arch="['Ampere']"
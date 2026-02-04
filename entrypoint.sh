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
SKIP_CLAUDE=${SKIP_CLAUDE:-false}  # 设为 true 可跳过 Claude 直接运行评估

echo "========================================================"
echo "🚀 Starting Claude Code Agent (AWS Bedrock Mode)"
echo "🎯 Target: Level $KB_LEVEL, Problem $KB_PROBLEM"
echo "📂 Output: runs/claude_code"
if [ "$SKIP_CLAUDE" = "true" ]; then
    echo "⚡ Mode: Evaluation Only (SKIP_CLAUDE=true)"
fi
echo "========================================================"

# --- 2. 构建动态 Prompt ---

AGENT_PROMPT=$(cat <<EOF
You are an expert CUDA engineer, specialized in writing high-performance GPU kernels on NVIDIA RTX A6000 (Ampere architecture).
Your task is to solve **Level $KB_LEVEL, Problem $KB_PROBLEM** in the KernelBench repository located in the current directory.
You have my approval to modify any files under the 'KernelBench' directory and run any scripts necessary to complete the task.

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

if [ "$SKIP_CLAUDE" = "true" ]; then
    echo ""
    echo "⚡ Skipping Claude Code (SKIP_CLAUDE=true)"
    echo "   Will directly evaluate existing code"
    echo ""
else
    echo ""
    echo ">>> Sending prompt to Claude Code..."
    echo ">>> Monitoring: tail -f /tmp/claude_output.log"
    echo ">>> To kill: pkill -f 'claude -p'"
    echo ""

    # 捕获中断信号
    trap 'echo ""; echo "🛑 Interrupted! Killing claude..."; pkill -9 -f "claude -p"; exit 130' INT TERM

    set +e  # 允许失败继续
    timeout 1800 claude -p "$AGENT_PROMPT" 2>&1 | tee /tmp/claude_output.log
    CLAUDE_EXIT=$?
    set -e

    trap - INT TERM  # 恢复默认信号处理

    echo ""
    echo "=========================================================="
    if [ $CLAUDE_EXIT -eq 0 ]; then
        echo "✅ Claude Code completed successfully"
    elif [ $CLAUDE_EXIT -eq 124 ]; then
        echo "⏱️  Claude Code timed out after 30 minutes"
    elif [ $CLAUDE_EXIT -eq 130 ]; then
        echo "🛑 Claude Code was interrupted"
    else
        echo "⚠️  Claude Code exited with code: $CLAUDE_EXIT"
    fi
    echo "=========================================================="
    echo ""
fi


uv run python scripts/eval_from_generations.py \
  run_name=example \
  dataset_src=local \
  level=1 \
  num_gpu_devices=1 \
  timeout=300 \
  subset="(1,1)" \
  gpu_arch="['Ampere']"
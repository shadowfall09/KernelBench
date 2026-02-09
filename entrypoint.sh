#!/bin/bash


KB_LEVEL=${KB_LEVEL:-1}
KB_PROBLEM=${KB_PROBLEM:-5}

echo "=========================================================="
echo "🚀 Starting Claude Code Agent"
echo "🎯 Target: Level $KB_LEVEL, Problem $KB_PROBLEM"
echo "📂 Output: runs/claude_code"
echo "=========================================================="


AGENT_PROMPT=$(cat <<EOF
You are an expert CUDA engineer, specialized in writing high-performance GPU kernels on NVIDIA RTX A6000 (Ampere architecture).
Your task is to solve **Level $KB_LEVEL, Problem $KB_PROBLEM** in the KernelBench repository located in the current directory.
You must write a CUDA kernel that is both correct and optimized for performance. If you are unable to optimize further in 5 rounds, provide a correct implementation.
Output your intermediate thoughts in real-time as you work through the problem.

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

Tips:
- You may use NVIDIA Nsight Compute (ncu) to profile and optimize your kernel. 
- You may use search to find optimization techniques for your specific problem.

Optimization Goal:
Iterate on the kernel until performance is maximized while maintaining correctness.
EOF
)

echo ""
echo ">>> Sending prompt to Claude Code..."

claude -p "$AGENT_PROMPT" --allowedTools "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell" --output-format stream-json --verbose --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'


# uv run python scripts/eval_from_generations.py \
#   run_name=example \
#   dataset_src=local \
#   level=1 \
#   num_gpu_devices=1 \
#   timeout=300 \
#   subset="(1,1)" \
#   gpu_arch="['Ampere']"
#!/bin/bash


KB_LEVEL=${KB_LEVEL:-1}
KB_PROBLEM=${KB_PROBLEM:-5}
KB_ENABLE_NCU=${KB_ENABLE_NCU:-true}
# Modal GPU type used for cloud evaluation (e.g. A10G, L40S, H100, A100)
KB_MODAL_GPU=${KB_MODAL_GPU:-A10G}

echo "=========================================================="
echo "🚀 Starting Claude Code Agent (Modal mode)"
echo "🎯 Target: Level $KB_LEVEL, Problem $KB_PROBLEM"
echo "☁️  Eval GPU: $KB_MODAL_GPU (via Modal)"
echo "📂 Output: runs/claude_code"
echo "=========================================================="


AGENT_PROMPT=$(cat <<EOF
You are an expert CUDA engineer, specialized in writing high-performance GPU kernels.
Your task is to solve **Level $KB_LEVEL, Problem $KB_PROBLEM** in the KernelBench repository.
**Target GPU**: $KB_MODAL_GPU
You must write a CUDA kernel that is both correct and optimized for performance. If you are unable to optimize further in 5 rounds, provide a correct implementation.
Output your intermediate thoughts in real-time as you work through the problem.

Implementation:
- Reference PyTorch implementations are in: KernelBench/ and an example solution is in: runs/example
- Write your solution under: runs/claude_code (mkdir if needed)
- Preserve the file naming pattern (level_X_problem_Y_sample_0_kernel.py).
- Write a correct and optimized CUDA implementation compatible with KernelBench.

Evaluation Command:
You can run the following command to verify your solution.
Note: The parameter \`subset="($KB_PROBLEM,$KB_PROBLEM)"\` explicitly tells the script to ONLY test Problem $KB_PROBLEM.
Note: eval_mode=modal offloads GPU execution to Modal cloud — no local GPU required.

uv run python scripts/eval_from_generations.py \\
  run_name=claude_code \\
  dataset_src=local \\
  level=$KB_LEVEL \\
  timeout=300 \\
  subset="($KB_PROBLEM,$KB_PROBLEM)" \\
  eval_mode=modal \\
  gpu=$KB_MODAL_GPU

Results will be written to: runs/claude_code/eval_results.json
Delete this file first if you need to evaluate again.

Tips:
- Do NOT run scripts/generate_samples.py. But you can read its logic to understand the required output format and conventions.
- You may use search to find optimization techniques for your specific problem.
EOF
)

if [ "$KB_ENABLE_NCU" = "true" ]; then
    AGENT_PROMPT+=$'\n- You may use NVIDIA Nsight Compute (ncu) to profile and optimize your kernel (available via Modal eval).'
fi

AGENT_PROMPT+=$'\n'
AGENT_PROMPT+=$'\n\nOptimization Goal:'
AGENT_PROMPT+=$'\nIterate on the kernel until performance is maximized while maintaining correctness.'

echo ""
echo ">>> Sending prompt to Claude Code..."

if [ "$KB_ENABLE_NCU" = "true" ]; then
    claude -p "$AGENT_PROMPT" --allowedTools "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell" --output-format stream-json --verbose --include-partial-messages | \
      jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
else
    claude -p "$AGENT_PROMPT" --allowedTools "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell" --disallowedTools "Bash(ncu *)" --output-format stream-json --verbose --include-partial-messages | \
      jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
fi

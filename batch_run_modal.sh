#!/bin/bash

# Modal version of batch_run.sh
#
# Key differences from batch_run.sh:
#   - Docker image has NO CUDA (eval is offloaded to Modal cloud GPUs)
#   - Parallel concurrency is controlled by NUM_PARALLEL (not GPU IDs)
#   - Step 1 (Evaluate) and Step 2 (Analysis) run locally, not inside Docker
#   - eval_from_generations.py is called with eval_mode=modal
#
# Usage:
#   Overwrite mode (default): ./batch_run_modal.sh --level 1 --start 30 --end 30 --run-name test_modal_local --parallel 4
#   Resume mode:              ./batch_run_modal.sh --level 1 --start 1 --end 25 --run-name my_run --parallel 8 --resume


# Get workspace directory (script location)
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# default parameters
LEVEL=1
START=1
END=10
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
NUM_PARALLEL=4                  # number of concurrent Docker containers (replaces CUDA_VISIBLE_DEVICES)
MODAL_GPU="H100"                # Modal cloud GPU type for evaluation (A10G, L40S, H100, A100, L4, T4)
AWS_PROFILE="bedrock"
DOCKER_IMAGE="kb-claude-modal:v1"
TIMEOUT=300
BASELINE="baseline_time_torch"
RESUME_MODE=false               # false=Overwrite mode, true=Resume mode (skip completed problems)
ENABLE_NCU=true                 # true=enable NVIDIA Nsight Compute profiling, false=disable NCU
MODEL_NAME="sonnet"             # sonnet or opus
ANTHROPIC_SMALL_FAST_MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level) LEVEL="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --parallel) NUM_PARALLEL="$2"; shift 2 ;;
        --gpu) MODAL_GPU="$2"; shift 2 ;;
        --profile) AWS_PROFILE="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        --resume) RESUME_MODE=true; shift 1 ;;
        --enable-ncu) ENABLE_NCU="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --small-model) ANTHROPIC_SMALL_FAST_MODEL="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --level LEVEL        KernelBench level (default: 1)"
            echo "  --start START        Starting problem number (default: 1)"
            echo "  --end END            Ending problem number (default: 10)"
            echo "  --run-name NAME      Run name (default: run_YYYYMMDD_HHMMSS)"
            echo "  --parallel N         Number of concurrent containers (default: 4)"
            echo "  --gpu GPU            Modal cloud GPU type for eval (default: A10G)"
            echo "                       Choices: A10G, L40S, H100, A100, A100-80GB, L4, T4"
            echo "  --profile PROFILE    AWS Profile (default: bedrock)"
            echo "  --timeout SECONDS    Evaluation timeout (default: 300)"
            echo "  --baseline BASELINE  Baseline name (default: baseline_time_torch)"
            echo "  --resume             Resume mode: skip already completed problems"
            echo "  --enable-ncu BOOL    Enable NCU profiling tool (default: true)"
            echo "  --model MODEL        Model: 'sonnet' or 'opus' (default: sonnet)"
            echo "  --small-model MODEL  Small fast model (default: claude-haiku-4-5)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Map MODEL_NAME to full ANTHROPIC_MODEL string
case "$MODEL_NAME" in
    sonnet)
        ANTHROPIC_MODEL="global.anthropic.claude-sonnet-4-6"
        # ANTHROPIC_MODEL="global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        ;;
    opus)
        ANTHROPIC_MODEL="global.anthropic.claude-opus-4-6-v1"
        ;;
    *)
        echo "Error: Invalid model name '$MODEL_NAME'. Must be 'sonnet' or 'opus'"
        exit 1
        ;;
esac

echo "========================================="
echo "KernelBench Modal batch run script (FIFO)"
echo "========================================="
echo "Level: $LEVEL"
echo "Problem range: $START - $END"
echo "Run name: $RUN_NAME"
echo "Model: $MODEL_NAME ($ANTHROPIC_MODEL)"
echo "Parallel containers: $NUM_PARALLEL"
echo "Modal eval GPU: $MODAL_GPU"
echo "Mode: $([ "$RESUME_MODE" = true ] && echo 'Resume (skip completed)' || echo 'Overwrite (run all)')"
echo "========================================="

# Directories
TEMP_OUTPUT_DIR="${WORKSPACE_DIR}/runs_output_temp_${RUN_NAME}"
FINAL_OUTPUT_DIR="${WORKSPACE_DIR}/runs/${RUN_NAME}"
LOG_DIR="${WORKSPACE_DIR}/logs_${RUN_NAME}"
mkdir -p "$TEMP_OUTPUT_DIR" "$FINAL_OUTPUT_DIR" "$LOG_DIR"

# Generate run configuration file
RUN_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
CONFIG_FILE="$FINAL_OUTPUT_DIR/run_config.json"
cat > "$CONFIG_FILE" << EOF
{
  "run_name": "$RUN_NAME",
  "start_time": "$RUN_START_TIME",
  "level": $LEVEL,
  "problem_range": {
    "start": $START,
    "end": $END
  },
  "num_parallel": $NUM_PARALLEL,
  "modal_gpu": "$MODAL_GPU",
  "timeout": $TIMEOUT,
  "baseline": "$BASELINE",
  "resume_mode": $RESUME_MODE,
  "enable_ncu": $ENABLE_NCU,
  "model_name": "$MODEL_NAME",
  "anthropic_model": "$ANTHROPIC_MODEL",
  "anthropic_small_fast_model": "$ANTHROPIC_SMALL_FAST_MODEL",
  "docker_image": "$DOCKER_IMAGE",
  "aws_profile": "$AWS_PROFILE",
  "output_directory": "$FINAL_OUTPUT_DIR",
  "log_directory": "$LOG_DIR",
  "platform": "modal"
}
EOF
echo "Configuration saved to: $CONFIG_FILE"

# Build Docker image (ONCE, blocking) — uses Dockerfile.modal (no CUDA)
echo "Building Docker image (no CUDA, eval via Modal)..."
docker build -f Dockerfile.modal -t $DOCKER_IMAGE . || exit 1

# ==============================================================================
#  CONCURRENCY CONTROL: NAMED PIPE (FIFO)
#  Tokens are parallel slot IDs (1..NUM_PARALLEL) instead of GPU IDs
# ==============================================================================
FIFO_FILE="/tmp/$$.fifo"
mkfifo "$FIFO_FILE"
exec 6<>"$FIFO_FILE"  # Link file descriptor 6 to the FIFO
rm "$FIFO_FILE"       # Remove file entry, FD remains open

# 1. Initialize tokens: Push slot IDs into the pipe
for ((slot=1; slot<=NUM_PARALLEL; slot++)); do
    echo "$slot" >&6
done

# Check if a problem is already completed
is_problem_completed() {
    local problem=$1
    local final_file="$FINAL_OUTPUT_DIR/level_${LEVEL}_problem_${problem}_sample_0_kernel.py"
    local temp_file="$TEMP_OUTPUT_DIR/${LEVEL}_${problem}/level_${LEVEL}_problem_${problem}_sample_0_kernel.py"

    # Check final directory first (preferred location)
    if [ -f "$final_file" ] && [ -s "$final_file" ]; then
        return 0
    fi

    # Check temp directory (if previous run was interrupted)
    if [ -f "$temp_file" ] && [ -s "$temp_file" ]; then
        echo "  → Found in temp, copying to final directory..."
        cp "$temp_file" "$FINAL_OUTPUT_DIR/" 2>/dev/null
        return 0
    fi

    return 1
}

run_task() {
    local problem=$1
    local slot=$2
    local log_file="$LOG_DIR/problem_${problem}_slot_${slot}.log"
    local problem_output_dir="${TEMP_OUTPUT_DIR}/${LEVEL}_${problem}"

    mkdir -p "$problem_output_dir"

    echo "[$(date '+%H:%M:%S')] Starting P${problem} on slot ${slot}..."

    # Run Docker — no GPU flag needed (container has no CUDA)
    docker run --rm \
        --cap-add=SYS_ADMIN \
        --security-opt seccomp=unconfined \
        -e AWS_PROFILE="$AWS_PROFILE" \
        -e KB_LEVEL=$LEVEL \
        -e KB_PROBLEM=$problem \
        -e KB_ENABLE_NCU="$ENABLE_NCU" \
        -e KB_MODAL_GPU="$MODAL_GPU" \
        -e ANTHROPIC_MODEL="$ANTHROPIC_MODEL" \
        -e ANTHROPIC_SMALL_FAST_MODEL="$ANTHROPIC_SMALL_FAST_MODEL" \
        -v "$HOME/.aws:/root/.aws:ro" \
        -v "$HOME/.modal.toml:/root/.modal.toml:ro" \
        -v "${problem_output_dir}:/app/KernelBench/runs/claude_code" \
        $DOCKER_IMAGE \
        /app/KernelBench/entrypoint_modal.sh > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] P${problem} DONE (slot ${slot}) ✓"
        cp "$problem_output_dir"/*.py "$FINAL_OUTPUT_DIR/" 2>/dev/null || true
    else
        echo "[$(date '+%H:%M:%S')] P${problem} FAILED (slot ${slot}) ✗"
    fi
}

# ==============================================================================
#  MAIN LOOP
# ==============================================================================
echo ""
echo "Starting parallel execution with $NUM_PARALLEL concurrent containers..."

# Count skipped tasks
skipped=0

for ((problem=START; problem<=END; problem++)); do
    # Check if in resume mode and problem is completed
    if [ "$RESUME_MODE" = true ] && is_problem_completed "$problem"; then
        echo "[$(date '+%H:%M:%S')] P${problem} already completed, skipping..."
        ((skipped++))
        continue
    fi

    # 2. Acquire a token (parallel slot)
    # This command BLOCKS until a slot becomes available
    read -u 6 slot_token

    # 3. Launch background job
    {
        # Execute the task
        run_task "$problem" "$slot_token"

        # 4. Return the token
        # ALWAYS execute this, even if run_task fails, so the slot isn't lost forever
        echo "$slot_token" >&6
    } &

    # Small delay to prevent race conditions on log creation
    sleep 0.5
done

# Wait for all background jobs to finish
wait

# Close FD
exec 6>&-

echo ""
echo "========================================="
echo "All tasks completed!"
if [ "$RESUME_MODE" = true ] && [ $skipped -gt 0 ]; then
    echo "Skipped (already done): $skipped"
    echo "Executed: $((END - START + 1 - skipped))"
fi
echo "========================================="

# Clean up
rm -rf "$TEMP_OUTPUT_DIR"
echo "Results saved to: $FINAL_OUTPUT_DIR"

# ==============================================================================
#  EVALUATION STAGE — runs locally (not in Docker), offloads GPU to Modal
# ==============================================================================

echo ""
echo "========================================="
echo "Starting Evaluation Stage (local → Modal cloud)..."
echo "========================================="

echo "Step 1: Evaluate via Modal ($MODAL_GPU)..."
uv run python scripts/eval_from_generations.py \
    run_name="${RUN_NAME}" \
    dataset_src=local \
    level=${LEVEL} \
    timeout=${TIMEOUT} \
    subset="(${START},${END})" \
    eval_mode=modal \
    gpu="${MODAL_GPU}"

echo ""
echo "Step 2: Analysis..."
uv run python scripts/benchmark_eval_analysis.py \
    run_name="${RUN_NAME}" \
    level=${LEVEL} \
    hardware=${MODAL_GPU}_Modal \
    baseline=${BASELINE}

echo ""
echo "========================================="
echo "Evaluation and Analysis Complete!"
echo "Results: ${WORKSPACE_DIR}/runs/${RUN_NAME}/"
echo "=========================================="

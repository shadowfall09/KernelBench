#!/bin/bash

# Usage: 
#   Overwrite mode (default): ./batch_run.sh --level 1 --start 1 --end 10 --run-name my_run --gpus "0,1,2,3"
#   Resume mode:              ./batch_run.sh --level 1 --start 1 --end 25 --run-name my_run --gpus "0,1,2,3,4,5,6,7" --resume

# 注意：不要在主流程开启 set -e，否则子进程报错可能导致主进程退出
# set -e 

# default parameters
LEVEL=1
START=1
END=10
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="0,1,2,3"
AWS_PROFILE="bedrock"
DOCKER_IMAGE="kb-claude:v1"
TIMEOUT=300
HARDWARE="RTX_A6000"
BASELINE="baseline_time_torch"
RESUME_MODE=false  # false=覆盖模式, true=继续运行模式

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level) LEVEL="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --gpus) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --profile) AWS_PROFILE="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --hardware) HARDWARE="$2"; shift 2 ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        --resume) RESUME_MODE=true; shift 1 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --level LEVEL        KernelBench level (default: 1)"
            echo "  --start START        Starting problem number (default: 1)"
            echo "  --end END            Ending problem number (default: 10)"
            echo "  --run-name NAME      Run name (default: run_YYYYMMDD_HHMMSS)"
            echo "  --gpus DEVICES       CUDA device list, comma-separated (default: 0,1,2,3)"
            echo "  --profile PROFILE    AWS Profile (default: bedrock)"
            echo "  --timeout SECONDS    Evaluation timeout (default: 300)"
            echo "  --hardware HW        Hardware name (default: RTX_A6000)"
            echo "  --baseline BASELINE  Baseline name (default: baseline_time_torch)"
            echo "  --resume             Resume mode: skip already completed problems"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Convert CUDA_VISIBLE_DEVICES to array
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "========================================="
echo "KernelBench PARALLEL batch run script (FIFO)"
echo "========================================="
echo "Level: $LEVEL"
echo "Problem range: $START - $END"
echo "Run name: $RUN_NAME"
echo "Available GPUs: ${GPU_ARRAY[@]} (Total $NUM_GPUS)"
echo "Mode: $([ "$RESUME_MODE" = true ] && echo 'Resume (skip completed)' || echo 'Overwrite (run all)')"
echo "========================================="

# Directories
TEMP_OUTPUT_DIR="$(pwd)/runs_output_temp_${RUN_NAME}"
FINAL_OUTPUT_DIR="/home/yichengtao/KernelBench/runs/${RUN_NAME}"
LOG_DIR="$(pwd)/logs_${RUN_NAME}"
mkdir -p "$TEMP_OUTPUT_DIR" "$FINAL_OUTPUT_DIR" "$LOG_DIR"

# Build Docker image (ONCE, blocking)
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE . || exit 1

# ==============================================================================
#  CONCURRENCY CONTROL: NAMED PIPE (FIFO)
# ==============================================================================
FIFO_FILE="/tmp/$$.fifo"
mkfifo "$FIFO_FILE"
exec 6<>"$FIFO_FILE"  # Link file descriptor 6 to the FIFO
rm "$FIFO_FILE"       # Remove file entry, FD remains open

# 1. Initialize tokens: Push each GPU ID into the pipe
for gpu in "${GPU_ARRAY[@]}"; do
    echo "$gpu" >&6
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
        # Copy to final directory if found in temp
        echo "  → Found in temp, copying to final directory..."
        cp "$temp_file" "$FINAL_OUTPUT_DIR/" 2>/dev/null
        return 0
    fi
    
    return 1
}

# Function needs to be exported or defined before use
run_task() {
    local problem=$1
    local gpu=$2
    local log_file="$LOG_DIR/problem_${problem}_gpu_${gpu}.log"
    local problem_output_dir="${TEMP_OUTPUT_DIR}/${LEVEL}_${problem}"
    
    mkdir -p "$problem_output_dir"
    
    echo "[$(date '+%H:%M:%S')] Starting P${problem} on GPU ${gpu}..."
    
    # Run Docker
    docker run --rm \
        --gpus "device=${gpu}" \
        --cap-add=SYS_ADMIN \
        --security-opt seccomp=unconfined \
        -e AWS_PROFILE="$AWS_PROFILE" \
        -e KB_LEVEL=$LEVEL \
        -e KB_PROBLEM=$problem \
        -v "$HOME/.aws:/root/.aws:ro" \
        -v "${problem_output_dir}:/app/KernelBench/runs/claude_code" \
        $DOCKER_IMAGE \
        /app/KernelBench/entrypoint.sh > "$log_file" 2>&1
        
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] P${problem} DONE (GPU ${gpu}) ✓"
        cp "$problem_output_dir"/*.py "$FINAL_OUTPUT_DIR/" 2>/dev/null || true
    else
        echo "[$(date '+%H:%M:%S')] P${problem} FAILED (GPU ${gpu}) ✗"
    fi
}

# ==============================================================================
#  MAIN LOOP
# ==============================================================================
echo ""
echo "Starting parallel execution across $NUM_GPUS GPUs..."

# Count skipped tasks
skipped=0

for ((problem=START; problem<=END; problem++)); do
    # Check if in resume mode and problem is completed
    if [ "$RESUME_MODE" = true ] && is_problem_completed "$problem"; then
        echo "[$(date '+%H:%M:%S')] P${problem} already completed, skipping..."
        ((skipped++))
        continue
    fi
    
    # 2. Acquire a token (GPU)
    # This command BLOCKS until a line (a GPU ID) is available to read
    read -u 6 gpu_token
    
    # 3. Launch background job
    {
        # Execute the task
        run_task "$problem" "$gpu_token"
        
        # 4. Return the token
        # ALWAYS execute this, even if run_task fails, so the GPU isn't lost forever
        echo "$gpu_token" >&6
    } & 
    
    # Don't sleep too long, just enough to prevent race conditions on log creation
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
#  EVALUATION STAGE (in Docker container)
# ==============================================================================

echo ""
echo "========================================="
echo "Starting Evaluation Stage (in Docker)..."
echo "========================================="

# For evaluation, use CUDA_VISIBLE_DEVICES to control which GPUs are visible
# This avoids Docker GPU allocation conflicts
GPU_DEVICES=$(IFS=,; echo "${GPU_ARRAY[*]}")

echo "Step 1: Evaluate..."
docker run --rm \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    -v "/home/yichengtao/KernelBench:/app/KernelBench" \
    -w /app/KernelBench \
    $DOCKER_IMAGE \
    bash -c "uv run python scripts/eval_from_generations.py \
        run_name='${RUN_NAME}' \
        dataset_src=local \
        level=${LEVEL} \
        num_gpu_devices=${NUM_GPUS} \
        timeout=${TIMEOUT} \
        subset='(${START},${END})' \
        gpu_arch=\"['Ampere']\""

echo ""
echo "Step 2: Analysis..."
docker run --rm \
    -v "/home/yichengtao/KernelBench:/app/KernelBench" \
    -w /app/KernelBench \
    $DOCKER_IMAGE \
    bash -c "uv run python scripts/benchmark_eval_analysis.py \
        run_name='${RUN_NAME}' \
        level=${LEVEL} \
        hardware=${HARDWARE} \
        baseline=${BASELINE}"

echo ""
echo "========================================="
echo "Evaluation and Analysis Complete!"
echo "Results: /home/yichengtao/KernelBench/runs/${RUN_NAME}/"
echo "========================================="
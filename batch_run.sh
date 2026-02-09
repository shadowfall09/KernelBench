#!/bin/bash

# usage: ./batch_run.sh --level 1 --start 1 --end 10 --run-name my_run --gpus "0,1,2,3"

set -e

# default parameters
LEVEL=1
START=1
END=10
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="0,1,2,3"
AWS_TOKEN="token"
DOCKER_IMAGE="kb-claude:v1"
TIMEOUT=300
HARDWARE="RTX_A6000"
BASELINE="baseline_time_torch"

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --gpus)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --token)
            AWS_TOKEN="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --hardware)
            HARDWARE="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --level LEVEL        KernelBench level (default: 1)"
            echo "  --start START        Starting problem number (default: 1)"
            echo "  --end END            Ending problem number (default: 10)"
            echo "  --run-name NAME      Run name (default: run_YYYYMMDD_HHMMSS)"
            echo "  --gpus DEVICES       CUDA device list, comma-separated (default: 0,1,2,3)"
            echo "  --token TOKEN        AWS Bearer Token (default: token)"
            echo "  --timeout SECONDS    Evaluation timeout (default: 300)"
            echo "  --hardware HW        Hardware name (default: RTX_A6000)"
            echo "  --baseline BASELINE  Baseline name (default: baseline_time_torch)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Convert CUDA_VISIBLE_DEVICES to array
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "========================================="
echo "KernelBench batch run script"
echo "========================================="
echo "Level: $LEVEL"
echo "Problem range: $START - $END"
echo "Run name: $RUN_NAME"
echo "Available GPUs: ${GPU_ARRAY[@]} (Total $NUM_GPUS)"
echo "Docker image: $DOCKER_IMAGE"
echo "========================================="

# Create temporary output directory
TEMP_OUTPUT_DIR="$(pwd)/runs_output_temp_${RUN_NAME}"
mkdir -p "$TEMP_OUTPUT_DIR"

# Create final output directory
FINAL_OUTPUT_DIR="/home/yichengtao/KernelBench/runs/${RUN_NAME}"
mkdir -p "$FINAL_OUTPUT_DIR"

# Build Docker image
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE .

# Create task queue
PROBLEMS=()
for ((i=START; i<=END; i++)); do
    PROBLEMS+=($i)
done

# Create log directory
LOG_DIR="$(pwd)/logs_${RUN_NAME}"
mkdir -p "$LOG_DIR"

# Background job arrays
declare -A RUNNING_JOBS
declare -A JOB_PROBLEMS

# Function: Run a single problem
run_problem() {
    local problem=$1
    local gpu=$2
    local log_file="$LOG_DIR/problem_${problem}_gpu_${gpu}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Problem $problem on GPU $gpu" | tee -a "$log_file"
    
    # Create a separate output directory for each problem
    local problem_output_dir="${TEMP_OUTPUT_DIR}/${LEVEL}_${problem}"
    mkdir -p "$problem_output_dir"
    
    docker run --rm \
        --gpus "device=${gpu}" \
        --cap-add=SYS_ADMIN \
        --security-opt seccomp=unconfined \
        -e AWS_BEARER_TOKEN_BEDROCK="$AWS_TOKEN" \
        -e KB_LEVEL=$LEVEL \
        -e KB_PROBLEM=$problem \
        -v "${problem_output_dir}:/app/KernelBench/runs/claude_code" \
        $DOCKER_IMAGE >> "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu completed Problem $problem ✓" | tee -a "$log_file"
        # Copy only Python files to final directory
        if [ -d "$problem_output_dir" ]; then
            cp "$problem_output_dir"/*.py "$FINAL_OUTPUT_DIR/" 2>/dev/null || true
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu failed Problem $problem (Exit code: $exit_code) ✗" | tee -a "$log_file"
    fi
    
    return $exit_code
}

# Main loop: Assign tasks to GPUs
problem_index=0
total_problems=${#PROBLEMS[@]}
completed=0
failed=0

echo ""
echo "Starting task allocation..."
echo ""

while [ $problem_index -lt $total_problems ] || [ ${#RUNNING_JOBS[@]} -gt 0 ]; do
    # Check completed tasks
    for gpu in "${!RUNNING_JOBS[@]}"; do
        pid=${RUNNING_JOBS[$gpu]}
        if ! kill -0 $pid 2>/dev/null; then
            # Task completed
            wait $pid
            exit_code=$?
            problem=${JOB_PROBLEMS[$gpu]}
            
            if [ $exit_code -eq 0 ]; then
                ((completed++))
            else
                ((failed++))
            fi
            
            unset RUNNING_JOBS[$gpu]
            unset JOB_PROBLEMS[$gpu]
            
            echo "Progress: $((completed + failed))/$total_problems (Success: $completed, Failed: $failed)"
        fi
    done
    
    # Assign new tasks to idle GPUs
    for gpu in "${GPU_ARRAY[@]}"; do
        if [ $problem_index -lt $total_problems ] && [ -z "${RUNNING_JOBS[$gpu]}" ]; then
            problem=${PROBLEMS[$problem_index]}
            run_problem $problem $gpu &
            pid=$!
            RUNNING_JOBS[$gpu]=$pid
            JOB_PROBLEMS[$gpu]=$problem
            ((problem_index++))
        fi
    done
    
    # Short sleep to avoid high CPU usage
    sleep 2
done

echo ""
echo "========================================="
echo "All tasks completed!"
echo "Success: $completed"
echo "Failed: $failed"
echo "Total: $total_problems"
echo "========================================="

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_OUTPUT_DIR"

echo ""
echo "Generated files have been saved to: $FINAL_OUTPUT_DIR"
echo ""

# Run evaluation
echo "========================================="
echo "Starting evaluation..."
echo "========================================="

cd /home/yichengtao/KernelBench

echo ""
echo "Step 1: Evaluate from generations..."
uv run python scripts/eval_from_generations.py \
    run_name="${RUN_NAME}" \
    dataset_src=local \
    level=$LEVEL \
    num_gpu_devices=$NUM_GPUS \
    timeout=$TIMEOUT

echo ""
echo "Step 2: Benchmark analysis..."
uv run python scripts/benchmark_eval_analysis.py \
    run_name="${RUN_NAME}" \
    level=$LEVEL \
    hardware=$HARDWARE \
    baseline=$BASELINE

echo ""
echo "========================================="
echo "Batch processing completed!"
echo "Run name: $RUN_NAME"
echo "Log directory: $LOG_DIR"
echo "Results directory: $FINAL_OUTPUT_DIR"
echo "========================================="

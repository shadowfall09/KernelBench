#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --gpus=8
#SBATCH --partition=taurus
#SBATCH --time=8:0:0
#SBATCH --account=yichengtao
#SBATCH --job-name=kernelbench
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Usage:
#   sbatch batch_run.sbatch
# Or with custom parameters:
#   sbatch --export=ALL,LEVEL=1,START=1,END=100,RUN_NAME=my_run batch_run.sbatch

# Set defaults if not provided via --export
LEVEL=${LEVEL:-1}
START=${START:-1}
END=${END:-100}
RUN_NAME=${RUN_NAME:-"run_$(date +%Y%m%d_%H%M%S)"}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
RESUME=${RESUME:-""}
HARDWARE=${HARDWARE:-"RTX_A6000"}
BASELINE=${BASELINE:-"baseline_time_torch"}
TIMEOUT=${TIMEOUT:-300}

echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo "Parameters:"
echo "  Level: $LEVEL"
echo "  Range: $START - $END"
echo "  Run name: $RUN_NAME"
echo "  GPUs: $GPUS"
echo "  Resume: $RESUME"
echo "========================================="

# Create logs directory if not exists
mkdir -p logs

# Change to KernelBench directory
cd /home/yichengtao/KernelBench || exit 1

# Build command line arguments
CMD_ARGS="--level $LEVEL --start $START --end $END --run-name $RUN_NAME --gpus $GPUS"
CMD_ARGS="$CMD_ARGS --hardware $HARDWARE --baseline $BASELINE --timeout $TIMEOUT"

if [ -n "$RESUME" ]; then
    CMD_ARGS="$CMD_ARGS --resume"
fi

# Execute batch_run.sh
echo "Executing: ./batch_run.sh $CMD_ARGS"
echo "Execution start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

START_TIME=$(date +%s)

./batch_run.sh $CMD_ARGS

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================="

exit $EXIT_CODE

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --job-name=kernelbench-verified
#SBATCH --output=slurm-%j.out
set -euo pipefail
# Submit from the checkout root. Set partition/account/gpu count using sbatch flags.
task_root="${KB_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$task_root"
task_gpu_count="${SLURM_GPUS_ON_NODE:-1}"
task_gpu_list="${GPUS:-$(seq -s, 0 "$((task_gpu_count - 1))") }"
task_gpu_list="${task_gpu_list// /}"
task_args=(--level "${LEVEL:-1}" --start "${START:-1}" --end "${END:-10}"
           --run-name "${RUN_NAME:-slurm_${SLURM_JOB_ID}}" --gpus "$task_gpu_list"
           --hardware "${HARDWARE:?Set HARDWARE to the verifier GPU label}"
           --gpu-arch "${GPU_ARCH:-Ampere}" --timeout "${TIMEOUT:-300}"
           --enable-ncu "${ENABLE_NCU:-true}" --resume "${RESUME:-false}")
if [[ -n "${BASELINE:-}" ]]; then task_args+=(--baseline "$BASELINE"); fi
if [[ -n "${AWS_PROFILE:-}" ]]; then task_args+=(--profile "$AWS_PROFILE"); fi
"${KB_PYTHON:-python3}" scripts/run_batch.py --generation "${GENERATION:-docker}" \
    --evaluation "${EVALUATION:-docker}" "${task_args[@]}"

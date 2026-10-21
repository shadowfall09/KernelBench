#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --job-name=kernelbench-pro
#SBATCH --output=slurm-%j.out
set -euo pipefail
# Submit from the checkout root. Set partition/account/gpu count using sbatch flags.
task_root="${KB_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$task_root"
task_gpu_count="${SLURM_GPUS_ON_NODE:-1}"
task_gpu_list="${GPUS:-$(seq -s, 0 "$((task_gpu_count - 1))") }"
task_gpu_list="${task_gpu_list// /}"
if [[ -n "${KB_CONFIG:-}" ]]; then
    task_args=(--config "$KB_CONFIG")
else
    task_args=(--level 1 --start 1 --end 10 --run-name "slurm_${SLURM_JOB_ID}"
               --gpus "$task_gpu_list" --generation docker --evaluation docker)
fi
# Only explicitly provided environment settings override a config file.
for task_setting in LEVEL:level START:start END:end RUN_NAME:run-name GPUS:gpus \
    HARDWARE:hardware GPU_ARCH:gpu-arch GPU_PROFILE:gpu-profile GPU:gpu TIMEOUT:timeout \
    ENABLE_NCU:enable-ncu RESUME:resume BASELINE:baseline AWS_PROFILE:profile \
    GENERATION:generation EVALUATION:evaluation; do
    task_variable="${task_setting%%:*}"
    if [[ -n "${!task_variable:-}" ]]; then
        task_args+=("--${task_setting#*:}" "${!task_variable}")
    fi
done
"${KB_PYTHON:-python3}" scripts/run_batch.py "${task_args[@]}"

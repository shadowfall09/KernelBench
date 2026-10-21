#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_defaults=(--generation docker --evaluation docker)
for task_argument in "$@"; do
    if [[ "$task_argument" == --config || "$task_argument" == --config=* ]]; then task_defaults=(); fi
done
exec "${KB_PYTHON:-python3}" "$task_root/scripts/run_batch.py" "${task_defaults[@]}" "$@"

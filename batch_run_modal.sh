#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${KB_PYTHON:-python3}" "$task_root/scripts/run_batch.py" --generation docker --evaluation modal "$@"

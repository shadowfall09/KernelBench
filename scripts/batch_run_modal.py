"""
KernelBench Batch Run using Modal Sandboxes

Replaces the Docker + batch_run.sh workflow with Modal's Sandbox API.
Each problem gets its own Modal Sandbox container with GPU access,
where Claude Code CLI agent generates and evaluates CUDA kernels.

Usage:
    # Basic usage
    uv run python scripts/batch_run_modal.py level=1 start=1 end=10 run_name=my_run

    # Full options
    uv run python scripts/batch_run_modal.py \
        level=1 start=1 end=2 run_name=test_modal gpu=A10G \
        resume=true enable_ncu=true

    # Resume interrupted run
    uv run python scripts/batch_run_modal.py level=1 start=1 end=25 run_name=my_run resume=true

Prerequisites:
    1. Modal account configured: `modal setup`
    2. AWS SSO login completed:
       aws sso login --profile=<your-profile-name>
       export AWS_PROFILE=<your-profile-name>
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
import pydra
from tqdm import tqdm

from batch_run_modal.config import BatchRunConfig, REPO_TOP_DIR
from batch_run_modal.agent import (
    run_single_problem_sandbox,
    is_problem_completed,
    download_kernels_from_volume,
    download_logs_from_volume,
)


# ==============================================================================
# Main
# ==============================================================================

@pydra.main(base=BatchRunConfig)
def main(config: BatchRunConfig):
    config.coerce_booleans()

    run_dir = os.path.join(REPO_TOP_DIR, "runs", config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("=========================================")
    print("KernelBench Modal Batch Run")
    print("=========================================")
    print(f"Level: {config.level}")
    print(f"Problem range: {config.start} - {config.end}")
    print(f"Run name: {config.run_name}")
    print(f"GPU: {config.gpu}")
    print(f"Parallel Sandboxes: {config.num_parallel}")
    print(f"Sandbox timeout: {config.sandbox_timeout}s")
    print(f"Mode: {'Resume (skip completed)' if config.resume else 'Overwrite (run all)'}")
    print(f"NCU: {'enabled' if config.enable_ncu else 'disabled'}")
    print(f"Optimization rounds: {config.optimization_rounds}")
    print("=========================================")

    # Save run config
    config_path = os.path.join(run_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "run_name": config.run_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": config.level,
            "problem_range": {"start": config.start, "end": config.end},
            "gpu": config.gpu,
            "num_parallel": config.num_parallel,
            "sandbox_timeout": config.sandbox_timeout,
            "baseline": config.baseline,
            "resume": config.resume,
            "enable_ncu": config.enable_ncu,
            "optimization_rounds": config.optimization_rounds,
            "aws_profile": config.aws_profile,
            "platform": "modal",
        }, f, indent=4)

    # ===== Phase 1: Generation via Modal Sandboxes =====
    print("\n[Phase 1] Generating kernels via Modal Sandboxes...")

    problems_to_run = []
    skipped = 0
    for pid in range(config.start, config.end + 1):
        if config.resume and is_problem_completed(run_dir, config.level, pid):
            print(f"  P{pid} already completed, skipping.")
            skipped += 1
            continue
        problems_to_run.append(pid)

    if not problems_to_run:
        print("All problems already completed!")
    else:
        print(f"Problems to solve: {len(problems_to_run)} (skipped: {skipped})")

        sandbox_app = modal.App.lookup("kb-claude-agent", create_if_missing=True)
        volume_name = f"kb-output-{config.run_name}"
        output_volume = modal.Volume.from_name(volume_name, create_if_missing=True)

        all_results = []

        with modal.enable_output():
            with ThreadPoolExecutor(max_workers=config.num_parallel) as executor:
                future_map = {}
                for pid in problems_to_run:
                    future = executor.submit(
                        run_single_problem_sandbox,
                        config.level, pid, config, output_volume, sandbox_app,
                    )
                    future_map[future] = pid

                with tqdm(total=len(problems_to_run), desc="Generation") as pbar:
                    for future in as_completed(future_map):
                        pid = future_map[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                        except Exception as e:
                            print(f"  [FATAL] P{pid}: {e}")
                            all_results.append({
                                "problem_id": pid,
                                "status": "fatal_error",
                                "error": str(e)[:500],
                            })
                        pbar.update(1)

        import time
        time.sleep(20)  # Wait for any final writes to the volume to complete
        print("\nDownloading generated kernels from Modal Volume...")
        download_kernels_from_volume(
            output_volume, run_dir, config.level, config.start, config.end
        )

        # Download agent logs from Volume
        print("\nDownloading agent logs from Modal Volume...")
        download_logs_from_volume(
            output_volume, run_dir, config.run_name, config.level, config.start, config.end
        )

        # Save generation summary
        summary = {
            "total": len(problems_to_run),
            "success": sum(1 for r in all_results if r["status"] == "success"),
            "failed": sum(1 for r in all_results if r["status"] == "failed"),
            "timeout": sum(1 for r in all_results if r["status"] == "timeout"),
            "error": sum(1 for r in all_results if r["status"] in ("error", "fatal_error")),
            "skipped": skipped,
            "details": all_results,
        }

        summary_path = os.path.join(run_dir, "generation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nGeneration Summary: {summary['success']}/{summary['total']} succeeded, "
              f"{summary['failed']} failed, {summary['timeout']} timed out, {summary['error']} errors")

    # ===== Phase 2: Evaluation =====
    if config.run_eval:
        print("\n[Phase 2] Running evaluation via eval_from_generations.py...")
        eval_cmd = [
            sys.executable,
            os.path.join(REPO_TOP_DIR, "scripts", "eval_from_generations.py"),
            f"run_name={config.run_name}",
            f"level={config.level}",
            f"subset=({config.start},{config.end})",
            "dataset_src=local",
            "eval_mode=modal",
            f"gpu={config.gpu}",
        ]
        subprocess.run(eval_cmd, check=True, cwd=REPO_TOP_DIR)

        # ===== Phase 3: Analysis =====
        print("\n[Phase 3] Running comparsion against baseline...")
        eval_results_path = os.path.join(run_dir, "eval_results.json")
        if not os.path.exists(eval_results_path):
            print("\nSkipping analysis — eval_results.json not found.")
            return
        
        analysis_cmd = [
            sys.executable,
            os.path.join(REPO_TOP_DIR, "scripts", "benchmark_eval_analysis.py"),
            f"run_name={config.run_name}",
            f"level={config.level}",
            f"hardware={config.gpu}",
            f"baseline={config.baseline}",
        ]

        print(f"Running: {' '.join(analysis_cmd)}")
        subprocess.run(analysis_cmd, cwd=REPO_TOP_DIR)

    print("\n=========================================")
    print("All stages complete!")
    print(f"Results: {run_dir}/")
    print("=========================================")


if __name__ == "__main__":
    main()

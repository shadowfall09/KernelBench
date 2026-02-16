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
        level=1 start=1 end=25 run_name=my_run \
        gpu=A10G num_parallel=8 sandbox_timeout=3600 \
        hardware=A10G baseline=baseline_time_torch \
        aws_profile=bedrock \
        resume=true enable_ncu=true

    # Resume interrupted run
    uv run python scripts/batch_run_modal.py level=1 start=1 end=25 run_name=my_run resume=true

Prerequisites:
    1. Modal account configured: `modal setup`
    2. AWS SSO login completed:
       aws sso login --profile=<your-profile-name>
       export AWS_PROFILE=<your-profile-name>
"""

import os
import sys
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
import pydra
from pydra import Config, REQUIRED
from tqdm import tqdm

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# Configuration
# ==============================================================================

class BatchRunConfig(Config):
    def __init__(self):
        self.level = REQUIRED                   # KernelBench level (1-4)
        self.start = 1                          # Starting problem ID
        self.end = 10                           # Ending problem ID
        self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Modal GPU type (A10G, L40S, H100, A100, A100-80GB, L4, T4)
        self.gpu = "A10G"
        # Number of parallel Sandboxes
        self.num_parallel = 8
        # Sandbox timeout in seconds (max time per problem)
        self.sandbox_timeout = 3600             # 1 hour

        # Evaluation settings
        self.eval_timeout = 300                 # Eval subprocess timeout
        self.hardware = "A10G"                  # Hardware name for analysis
        self.baseline = "baseline_time_torch"   # Baseline name for analysis

        # Resume mode: skip already completed problems
        self.resume = False
        # Enable NVIDIA Nsight Compute profiling
        self.enable_ncu = True

        # AWS profile name (used with `aws sso login --profile=xxx`)
        self.aws_profile = "bedrock"

        # Anthropic model config
        self.anthropic_model = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.anthropic_small_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

        # Whether to run evaluation + analysis after generation
        self.run_eval = True

    def __repr__(self):
        return f"BatchRunConfig({self.to_dict()})"


# ==============================================================================
# GPU Architecture Mapping
# ==============================================================================

GPU_ARCH_MAPPING = {
    "A10G": "Ampere",
    "A100": "Ampere",
    "A100-80GB": "Ampere",
    "L40S": "Ada",
    "L4": "Ada",
    "H100": "Hopper",
    "T4": "Turing",
}


# ==============================================================================
# Modal Image Definition
# ==============================================================================

SRC_DIR = os.path.join(REPO_TOP_DIR, "src")
KERNELBENCH_DIR = os.path.join(REPO_TOP_DIR, "KernelBench")
SCRIPTS_DIR = os.path.join(REPO_TOP_DIR, "scripts")

# Base evaluation image — shared foundation for both eval and agent
eval_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .uv_sync(uv_project_dir=REPO_TOP_DIR, extras=["gpu"])
    .env({"PYTHONPATH": "/root/src:/root"})
    .add_local_dir(SRC_DIR, remote_path="/root/src")
    .add_local_dir(KERNELBENCH_DIR, remote_path="/root/KernelBench")
)

# Agent image = eval_image + Claude CLI + jq/curl/vim (for agent interaction)
agent_image = (
    eval_image
    .apt_install("jq", "curl", "vim", "less")
    .run_commands(
        # Install Claude CLI and symlink to standard PATH
        "curl -fsSL https://claude.ai/install.sh | bash",
        "ln -sf /root/.anthropic/bin/claude /usr/local/bin/claude",
        # Create output directories
        "mkdir -p /root/runs/claude_code /root/cache",
    )
    .add_local_dir(SCRIPTS_DIR, remote_path="/root/scripts")
)


# ==============================================================================
# Modal App for Evaluation
# ==============================================================================

eval_app = modal.App("kb-batch-eval")


@eval_app.cls(
    image=eval_image,
    gpu="A10G",
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=1.0),
)
class ModalEvaluator:
    @modal.method()
    def evaluate_single_sample(
        self,
        ref_arch_src: str,
        kernel_src: str,
        gpu_arch: list[str],
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        measure_performance: bool = True,
        timing_method: str = "cuda_event",
        verbose: bool = False,
        backend: str = "cuda",
        precision: str = "fp32",
    ):
        from kernelbench.eval import eval_kernel_against_ref, KernelExecResult, get_torch_dtype_from_string
        from kernelbench.utils import set_gpu_arch
        import torch
        import time as _time
        import modal.experimental

        # Wait for GPU
        max_wait = 30
        t0 = _time.time()
        while _time.time() - t0 < max_wait:
            if torch.cuda.is_available():
                break
            _time.sleep(min(0.5 * (2 ** int((_time.time() - t0) / 2)), 8.0))
        else:
            raise RuntimeError(f"GPU not attached after {max_wait}s")

        set_gpu_arch(gpu_arch)

        gpu_corrupted = False
        try:
            result = eval_kernel_against_ref(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                measure_performance=measure_performance,
                timing_method=timing_method,
                verbose=verbose,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                build_dir=None,
                device=torch.device("cuda:0"),
                backend=backend,
                precision=get_torch_dtype_from_string(precision),
            )
        except (torch.cuda.CudaError, torch.AcceleratorError) as e:
            gpu_corrupted = True
            modal.experimental.stop_fetching_inputs()
            result = KernelExecResult(
                compiled=False, correctness=False,
                metadata={"gpu_error": type(e).__name__, "error_message": str(e)[:500]},
                runtime=-1.0, runtime_stats={},
            )

        if not gpu_corrupted:
            torch.cuda.empty_cache()

        return result


# ==============================================================================
# Agent Prompt Builder
# ==============================================================================

def build_agent_prompt(level: int, problem_id: int, gpu_arch: str = "Ampere", enable_ncu: bool = True) -> str:
    """Build the Claude Code agent prompt — same logic as entrypoint.sh"""
    prompt = f"""You are an expert CUDA engineer, specialized in writing high-performance GPU kernels.
Your task is to solve **Level {level}, Problem {problem_id}** in the KernelBench repository located in the current directory.
You must write a CUDA kernel that is both correct and optimized for performance. If you are unable to optimize further in 5 rounds, provide a correct implementation.
Output your intermediate thoughts in real-time as you work through the problem.

Rules:
1. Do NOT run scripts/generate_samples.py.
2. You may read its logic to understand the required output format and conventions.

Implementation:
- Reference implementations are in: KernelBench/
- Write your solution under: runs/claude_code
- Preserve the file naming pattern (level_X_problem_Y_sample_0_kernel.py).
- Write a correct and optimized CUDA implementation compatible with KernelBench.

Evaluation Command:
You MUST run the following command to verify your solution.
Note: The parameter `subset="({problem_id},{problem_id})"` explicitly tells the script to ONLY test Problem {problem_id}.

python scripts/eval_from_generations.py \\
  run_name=claude_code \\
  dataset_src=local \\
  level={level} \\
  num_gpu_devices=1 \\
  timeout=300 \\
  subset="({problem_id},{problem_id})" \\
  gpu_arch="['{gpu_arch}']"

Results will be written to: runs/claude_code/eval_results.json
Delete this file first if you need to evaluate again.

Tips:"""

    if enable_ncu:
        prompt += "\n- You may use NVIDIA Nsight Compute (ncu) to profile and optimize your kernel."

    prompt += "\n- You may use search to find optimization techniques for your specific problem."
    prompt += "\n\nOptimization Goal:"
    prompt += "\nIterate on the kernel until performance is maximized while maintaining correctness."

    return prompt


# ==============================================================================
# Core Functions
# ==============================================================================

def is_problem_completed(run_dir: str, level: int, problem_id: int) -> bool:
    """Check if a problem already has a generated kernel file."""
    kernel_path = os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
    )
    return os.path.exists(kernel_path) and os.path.getsize(kernel_path) > 0


def _sync_aws_dir_to_sandbox(sb: modal.Sandbox):
    """
    Copy the local ~/.aws directory into the sandbox at /root/.aws.
    This syncs SSO tokens, config, credentials, and cache files so that
    `aws` / Claude Code CLI can authenticate via the same SSO profile.
    """
    aws_dir = os.path.expanduser("~/.aws")
    if not os.path.isdir(aws_dir):
        raise FileNotFoundError(
            f"~/.aws directory not found. Run `aws sso login --profile=<profile>` first."
        )

    # Create the target directory structure
    sb.exec("mkdir", "-p", "/root/.aws").wait()

    for root, dirs, files in os.walk(aws_dir):
        rel_root = os.path.relpath(root, aws_dir)
        remote_root = f"/root/.aws/{rel_root}" if rel_root != "." else "/root/.aws"

        # Create subdirectories
        for d in dirs:
            remote_dir = f"{remote_root}/{d}"
            sb.exec("mkdir", "-p", remote_dir).wait()

        # Write files
        for fname in files:
            local_path = os.path.join(root, fname)
            remote_path = f"{remote_root}/{fname}"
            try:
                with open(local_path, "r") as lf:
                    content = lf.read()
                with sb.open(remote_path, "w") as rf:
                    rf.write(content)
            except (UnicodeDecodeError, PermissionError):
                # Binary file or no permission — read as bytes
                try:
                    with open(local_path, "rb") as lf:
                        content = lf.read()
                    with sb.open(remote_path, "wb") as rf:
                        rf.write(content)
                except Exception:
                    pass  # Skip files that can't be read


def run_single_problem_sandbox(
    level: int,
    problem_id: int,
    config: BatchRunConfig,
    output_volume: modal.Volume,
    app: modal.App,
) -> dict:
    """
    Launch a Modal Sandbox to solve one KernelBench problem using Claude Code CLI.
    Returns a dict with problem_id, status, and timing info.
    """
    start_time = time.time()
    result = {"problem_id": problem_id, "status": "unknown", "elapsed": 0}

    try:
        # Claude/Bedrock environment variables
        claude_env = {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "ANTHROPIC_MODEL": config.anthropic_model,
            "ANTHROPIC_SMALL_FAST_MODEL": config.anthropic_small_model,
            "AWS_PROFILE": config.aws_profile,
            "KB_LEVEL": str(level),
            "KB_PROBLEM": str(problem_id),
            "KB_ENABLE_NCU": str(config.enable_ncu).lower(),
        }

        # Create the Sandbox
        sb = modal.Sandbox.create(
            app=app,
            image=agent_image,
            gpu=config.gpu,
            env=claude_env,
            volumes={"/output": output_volume},
            timeout=config.sandbox_timeout,
            workdir="/root",
            cpu=4.0,
            memory=32768,  # 32 GB
        )

        print(f"  [Sandbox] P{problem_id} created: {sb.object_id}")

        # Sync local ~/.aws directory into the sandbox (SSO tokens, config, etc.)
        _sync_aws_dir_to_sandbox(sb)

        # Build the Claude Code agent prompt
        gpu_arch = GPU_ARCH_MAPPING.get(config.gpu, "Ampere")
        agent_prompt = build_agent_prompt(level, problem_id, gpu_arch, config.enable_ncu)

        # Write prompt to a file in the sandbox to avoid shell escaping issues
        with sb.open("/tmp/agent_prompt.txt", "w") as pf:
            pf.write(agent_prompt)

        # Construct the Claude Code CLI command (reads prompt from file)
        allowed_tools = "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell"
        if config.enable_ncu:
            claude_cmd = (
                f'claude -p "$(cat /tmp/agent_prompt.txt)" '
                f'--allowedTools "{allowed_tools}" '
                f'--output-format stream-json --verbose --include-partial-messages '
                f'2>&1 | jq -rj \'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text\' || true'
            )
        else:
            claude_cmd = (
                f'claude -p "$(cat /tmp/agent_prompt.txt)" '
                f'--allowedTools "{allowed_tools}" '
                f'--disallowedTools "Bash(ncu *)" '
                f'--output-format stream-json --verbose --include-partial-messages '
                f'2>&1 | jq -rj \'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text\' || true'
            )

        # Run Claude Code agent inside the sandbox
        proc = sb.exec(
            "bash", "-c", claude_cmd,
            timeout=config.sandbox_timeout - 60,  # leave margin
        )

        # Stream output (optional: could just wait)
        for line in proc.stdout:
            pass  # Consume stdout to prevent blocking

        proc.wait()

        # Copy generated kernel files to volume for persistence
        copy_cmd = (
            f"cp /root/runs/claude_code/level_{level}_problem_{problem_id}_*.py "
            f"/output/ 2>/dev/null; echo $?"
        )
        cp_proc = sb.exec("bash", "-c", copy_cmd)
        cp_output = cp_proc.stdout.read().strip()

        sb.terminate()

        elapsed = time.time() - start_time
        result["elapsed"] = elapsed

        # Check if kernel was generated
        if cp_output == "0":
            result["status"] = "success"
            print(f"  [OK] P{problem_id} completed in {elapsed:.0f}s")
        else:
            result["status"] = "failed"
            print(f"  [FAIL] P{problem_id} - no kernel generated ({elapsed:.0f}s)")

    except modal.exception.SandboxTimeoutError:
        elapsed = time.time() - start_time
        result["status"] = "timeout"
        result["elapsed"] = elapsed
        print(f"  [TIMEOUT] P{problem_id} after {elapsed:.0f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        result["status"] = "error"
        result["elapsed"] = elapsed
        result["error"] = str(e)[:500]
        print(f"  [ERROR] P{problem_id}: {str(e)[:200]}")

    return result


def download_kernels_from_volume(
    output_volume: modal.Volume, run_dir: str, level: int, start: int, end: int
):
    """Download generated kernel files from Modal Volume to local disk."""
    os.makedirs(run_dir, exist_ok=True)

    # Reload volume to get latest data
    output_volume.reload()

    downloaded = 0
    for problem_id in range(start, end + 1):
        filename = f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
        try:
            # Read file from volume (read_file returns an iterator of byte chunks)
            filepath = os.path.join(run_dir, filename)
            with open(filepath, "wb") as f:
                for data in output_volume.read_file(filename):
                    f.write(data)
            downloaded += 1
        except Exception:
            pass  # File doesn't exist for this problem

    print(f"Downloaded {downloaded} kernel files to {run_dir}")
    return downloaded


def run_evaluation_modal(config: BatchRunConfig, run_dir: str):
    """
    Run batch evaluation using Modal GPU containers.
    Uses the same ModalEvaluator pattern as eval_from_generations.py.
    """
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.utils import read_file

    print("\n=========================================")
    print("Starting Evaluation Stage (Modal)...")
    print("=========================================")

    dataset = construct_kernelbench_dataset(
        level=config.level,
        source="local",
    )

    gpu_arch = [GPU_ARCH_MAPPING.get(config.gpu, "Ampere")]
    eval_results = {}

    # Collect work items
    work_items = []
    for problem_id in range(config.start, config.end + 1):
        kernel_path = os.path.join(
            run_dir, f"level_{config.level}_problem_{problem_id}_sample_0_kernel.py"
        )
        if not os.path.exists(kernel_path):
            continue

        problem = dataset.get_problem_by_id(problem_id)
        ref_arch_src = problem.code
        kernel_src = read_file(kernel_path)

        if kernel_src:
            work_items.append({
                "problem_id": problem_id,
                "ref_arch_src": ref_arch_src,
                "kernel_src": kernel_src,
            })

    if not work_items:
        print("No kernels found to evaluate.")
        return

    print(f"Evaluating {len(work_items)} kernels on {config.gpu} GPUs...")

    with eval_app.run():
        evaluator_cls = (
            ModalEvaluator.with_options(gpu=config.gpu)
            if config.gpu != "A10G"
            else ModalEvaluator
        )

        batch_size = config.num_parallel
        remaining = list(work_items)

        with tqdm(total=len(work_items), desc="Eval Progress") as pbar:
            while remaining:
                batch = remaining[:batch_size]
                remaining = remaining[batch_size:]

                # Spawn parallel evaluations
                futures = []
                for item in batch:
                    future = evaluator_cls().evaluate_single_sample.spawn(
                        ref_arch_src=item["ref_arch_src"],
                        kernel_src=item["kernel_src"],
                        gpu_arch=gpu_arch,
                        num_correct_trials=5,
                        num_perf_trials=100,
                        measure_performance=True,
                        timing_method="cuda_event",
                        verbose=False,
                        backend="cuda",
                        precision="fp32",
                    )
                    futures.append((item["problem_id"], future))

                # Collect results
                for problem_id, future in futures:
                    try:
                        result = future.get(timeout=config.eval_timeout + 60)
                        eval_results[str(problem_id)] = [{
                            "sample_id": 0,
                            "compiled": result.compiled,
                            "correctness": result.correctness,
                            "metadata": result.metadata if hasattr(result, 'metadata') else {},
                            "runtime": result.runtime if hasattr(result, 'runtime') else -1.0,
                            "runtime_stats": result.runtime_stats if hasattr(result, 'runtime_stats') else {},
                        }]
                    except Exception as e:
                        print(f"  [EVAL ERROR] P{problem_id}: {str(e)[:200]}")
                        eval_results[str(problem_id)] = [{
                            "sample_id": 0,
                            "compiled": False,
                            "correctness": False,
                            "metadata": {"error": str(e)[:500]},
                            "runtime": -1.0,
                            "runtime_stats": {},
                        }]

                pbar.update(len(batch))

    # Save eval results
    eval_results_path = os.path.join(run_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {eval_results_path}")


def run_analysis(config: BatchRunConfig, run_dir: str):
    """Run benchmark analysis (purely local, no GPU needed)."""
    print("\n=========================================")
    print("Starting Analysis Stage...")
    print("=========================================")

    import subprocess
    analysis_cmd = [
        sys.executable, os.path.join(REPO_TOP_DIR, "scripts", "benchmark_eval_analysis.py"),
        f"run_name={config.run_name}",
        f"level={config.level}",
        f"hardware={config.hardware}",
        f"baseline={config.baseline}",
    ]

    print(f"Running: {' '.join(analysis_cmd)}")
    subprocess.run(analysis_cmd, cwd=REPO_TOP_DIR)


# ==============================================================================
# Main
# ==============================================================================

@pydra.main(base=BatchRunConfig)
def main(config: BatchRunConfig):
    # Handle string booleans from CLI
    if isinstance(config.resume, str):
        config.resume = config.resume.lower() in ["true", "1", "yes"]
    if isinstance(config.enable_ncu, str):
        config.enable_ncu = config.enable_ncu.lower() in ["true", "1", "yes"]
    if isinstance(config.run_eval, str):
        config.run_eval = config.run_eval.lower() in ["true", "1", "yes"]

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
            "hardware": config.hardware,
            "baseline": config.baseline,
            "resume": config.resume,
            "enable_ncu": config.enable_ncu,
            "aws_profile": config.aws_profile,
            "platform": "modal",
        }, f, indent=4)

    # ===== Phase 1: Generation via Modal Sandboxes =====
    print("\n[Phase 1] Generating kernels via Modal Sandboxes...")

    # Build work queue
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

        # Create or get a Modal App for Sandboxes
        sandbox_app = modal.App.lookup("kb-claude-agent", create_if_missing=True)

        # Create a Modal Volume to collect output files
        volume_name = f"kb-output-{config.run_name}"
        output_volume = modal.Volume.from_name(volume_name, create_if_missing=True)

        # Process problems in parallel batches using ThreadPoolExecutor
        all_results = []

        with modal.enable_output():
            with ThreadPoolExecutor(max_workers=config.num_parallel) as executor:
                future_map = {}
                for pid in problems_to_run:
                    future = executor.submit(
                        run_single_problem_sandbox,
                        config.level,
                        pid,
                        config,
                        output_volume,
                        sandbox_app,
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

        # Download kernel files from Volume
        print("\nDownloading generated kernels from Modal Volume...")
        download_kernels_from_volume(
            output_volume, run_dir, config.level, config.start, config.end
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
        run_evaluation_modal(config, run_dir)

        # ===== Phase 3: Analysis =====
        run_analysis(config, run_dir)

    print("\n=========================================")
    print("All stages complete!")
    print(f"Results: {run_dir}/")
    print("=========================================")


if __name__ == "__main__":
    main()

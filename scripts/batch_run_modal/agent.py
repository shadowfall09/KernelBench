"""
Claude Code agent sandbox management for KernelBench.

Provides:
    - build_agent_prompt()            : construct the prompt sent to Claude Code CLI
    - run_single_problem_sandbox()    : launch one Modal Sandbox for a single problem
    - is_problem_completed()          : check whether a kernel file already exists
    - download_kernels_from_volume()  : pull generated kernels from Modal Volume to local
"""

import os
import time

import modal

from .config import BatchRunConfig, GPU_ARCH_MAPPING


# ==============================================================================
# Agent Prompt Builder
# ==============================================================================

def build_agent_prompt(
    level: int,
    problem_id: int,
    gpu_arch: str = "Ampere",
    enable_ncu: bool = True,
) -> str:
    """Build the Claude Code agent prompt — same logic as entrypoint.sh."""
    prompt = f"""You are an expert CUDA engineer, specialized in writing high-performance GPU kernels for a specific GPU.
Your task is to solve **Level {level}, Problem {problem_id}** in the KernelBench repository located in the current directory.
You must write a CUDA kernel that is both correct and optimized for performance. If you are unable to optimize further in limited (e.g. 5) rounds, provide a correct implementation.
Output your intermediate thoughts in real-time as you work through the problem.

Implementation:
- Reference implementations are in: KernelBench/
- Write your solution under: runs/claude_code (mkdir if needed)
- Preserve the file naming pattern (level_X_problem_Y_sample_0_kernel.py).
- Write a correct and optimized CUDA implementation compatible with KernelBench.

Evaluation Command:
You can run the following command to verify your solution.
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

Tips:
- The Python environment you can use is located in `/.uv/.venv`.
- Do NOT run scripts/generate_samples.py. But you can read its logic to understand the required output format and conventions.
- Consider GPU architecture-specific optimizations, you may run `nvidia-smi` to check the GPU model and utilization.
- You may use search to find optimization techniques for your specific problem."""

    if enable_ncu:
        prompt += "\n- You may use NVIDIA Nsight Compute (ncu) to profile and optimize your kernel."

    prompt += "\n"
    prompt += "\n\nOptimization Goal:"
    prompt += "\nIterate on the kernel until performance is maximized while maintaining correctness."

    return prompt


# ==============================================================================
# Helpers
# ==============================================================================

def is_problem_completed(run_dir: str, level: int, problem_id: int) -> bool:
    """Check if a problem already has a generated kernel file."""
    kernel_path = os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
    )
    return os.path.exists(kernel_path) and os.path.getsize(kernel_path) > 0


def _build_claude_cmd(enable_ncu: bool) -> str:
    """Return the bash command that invokes Claude Code CLI inside the sandbox."""
    allowed_tools = "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell"
    base = (
        'claude -p "$(cat /tmp/agent_prompt.txt)" '
        f'--allowedTools "{allowed_tools}" '
    )
    if not enable_ncu:
        base += '--disallowedTools "Bash(ncu *)" '

    return base


def _run_sandbox_preflight(sb: modal.Sandbox, problem_id: int) -> str | None:
    """
    Run quick checks inside the sandbox before launching claude.
    Returns an error message string if something is wrong, None if OK.
    """
    # Check claude CLI exists
    check = sb.exec("bash", "-c", "which claude && claude --version 2>&1 || echo 'CLAUDE_NOT_FOUND'")
    out = check.stdout.read().strip()
    check.wait()
    if "CLAUDE_NOT_FOUND" in out or not out:
        return f"Claude CLI not found in sandbox: {out}"
    print(f"  [Preflight] P{problem_id} claude: {out.splitlines()[-1]}")

    # Check AWS credentials directory is mounted
    aws_check = sb.exec("bash", "-c",
        'test -d /root/.aws && echo "AWS_OK" || echo "AWS_MISSING"'
    )
    aws_out = aws_check.stdout.read().strip()
    aws_check.wait()
    if "AWS_OK" not in aws_out:
        return "AWS credentials directory not found in sandbox (/root/.aws)"
    print(f"  [Preflight] P{problem_id} AWS mount OK")

    return None


# ==============================================================================
# Sandbox Runner
# ==============================================================================

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
    from .images import agent_image  # deferred to avoid circular import at module level

    start_time = time.time()
    result = {"problem_id": problem_id, "status": "unknown", "elapsed": 0}

    try:
        # AWS credentials are now included in the agent_image
        claude_env = {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
            "AWS_PROFILE": "bedrock",
            "ANTHROPIC_MODEL": config.anthropic_model,
            "ANTHROPIC_SMALL_FAST_MODEL": config.anthropic_small_model,
            "KB_LEVEL": str(level),
            "KB_PROBLEM": str(problem_id),
            "KB_ENABLE_NCU": str(config.enable_ncu).lower(),
        }

        sb = modal.Sandbox.create(
            app=app,
            image=agent_image,
            gpu=config.gpu,
            env=claude_env,
            volumes={"/output": output_volume},
            timeout=config.sandbox_timeout,
            workdir="/root",
            cpu=4.0,
            memory=32768,
        )

        print(f"  [Sandbox] P{problem_id} created: {sb.object_id}")

        # --- Preflight checks ---
        preflight_err = _run_sandbox_preflight(sb, problem_id)
        if preflight_err:
            sb.terminate()
            elapsed = time.time() - start_time
            result["status"] = "error"
            result["elapsed"] = elapsed
            result["error"] = preflight_err
            print(f"  [PREFLIGHT FAIL] P{problem_id}: {preflight_err[:300]}")
            return result

        # Build & write prompt
        gpu_arch = GPU_ARCH_MAPPING.get(config.gpu, "Ampere")
        agent_prompt = build_agent_prompt(level, problem_id, gpu_arch, config.enable_ncu)
        with sb.open("/tmp/agent_prompt.txt", "w") as pf:
            pf.write(agent_prompt)

        # Run Claude Code
        claude_cmd = _build_claude_cmd(config.enable_ncu)
        proc = sb.exec(
            "bash", "-c", claude_cmd, pty=True,
            timeout=config.sandbox_timeout - 60,
        )

        # Capture agent stdout
        agent_output_lines = []
        for line in proc.stdout:
            agent_output_lines.append(line)
        proc.wait()

        agent_output = "".join(agent_output_lines).strip()

        # Retrieve stderr log
        stderr_proc = sb.exec("bash", "-c", "cat /tmp/claude_stderr.log 2>/dev/null | tail -50")
        stderr_content = stderr_proc.stdout.read().strip()
        stderr_proc.wait()

        # Print debug info
        if agent_output:
            preview = agent_output[-300:] if len(agent_output) > 300 else agent_output
            print(f"  [Agent] P{problem_id} output tail:\n{preview}")
        else:
            print(f"  [Agent] P{problem_id} produced NO output")

        if stderr_content:
            print(f"  [Agent] P{problem_id} stderr tail:\n{stderr_content[-500:]}")

        # Create logs directory and save agent output
        logs_dir = f"/output/logs_{config.run_name}"
        sb.exec("bash", "-c", f"mkdir -p {logs_dir}").wait()

        output_filename = f"{logs_dir}/level_{level}_problem_{problem_id}_sample_0_output.txt"
        with sb.open(output_filename, "w") as out_f:
            out_f.write(agent_output)

        # Also save stderr if present
        if stderr_content:
            stderr_filename = f"{logs_dir}/level_{level}_problem_{problem_id}_sample_0_stderr.txt"
            with sb.open(stderr_filename, "w") as err_f:
                err_f.write(stderr_content)

        # Copy kernel files to output volume
        copy_cmd = (
            f"cp /root/runs/claude_code/level_{level}_problem_{problem_id}_*.py "
            f"/output/ 2>/dev/null; echo $?"
        )
        cp_proc = sb.exec("bash", "-c", copy_cmd)
        cp_output = cp_proc.stdout.read().strip()

        # On failure, show what's actually in runs/claude_code/
        if cp_output != "0":
            ls_proc = sb.exec("bash", "-c",
                "echo '--- /root/runs/claude_code/ ---' && "
                "ls -la /root/runs/claude_code/ 2>&1 && "
                "echo '--- /root/ ---' && "
                "ls /root/ 2>&1"
            )
            ls_out = ls_proc.stdout.read().strip()
            ls_proc.wait()
            print(f"  [Debug] P{problem_id} file listing:\n{ls_out[:500]}")

        sb.terminate()

        elapsed = time.time() - start_time
        result["elapsed"] = elapsed

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


# ==============================================================================
# Volume Download
# ==============================================================================

def download_kernels_from_volume(
    output_volume: modal.Volume,
    run_dir: str,
    level: int,
    start: int,
    end: int,
) -> int:
    """Download generated kernel files from Modal Volume to local disk."""
    os.makedirs(run_dir, exist_ok=True)

    downloaded = 0
    for problem_id in range(start, end + 1):
        filename = f"level_{level}_problem_{problem_id}_sample_0_kernel.py"
        try:
            filepath = os.path.join(run_dir, filename)
            with open(filepath, "wb") as f:
                for data in output_volume.read_file(filename):
                    f.write(data)
            downloaded += 1
        except Exception:
            pass

    print(f"Downloaded {downloaded} kernel files to {run_dir}")
    return downloaded


def download_logs_from_volume(
    output_volume: modal.Volume,
    run_dir: str,
    run_name: str,
    level: int,
    start: int,
    end: int,
) -> int:
    """Download agent logs from Modal Volume to local disk."""
    logs_dir = os.path.join(run_dir, f"logs_{run_name}")
    os.makedirs(logs_dir, exist_ok=True)

    downloaded = 0
    for problem_id in range(start, end + 1):
        # Download output log
        output_filename = f"logs_{run_name}/level_{level}_problem_{problem_id}_sample_0_output.txt"
        try:
            filepath = os.path.join(logs_dir, f"level_{level}_problem_{problem_id}_sample_0_output.txt")
            with open(filepath, "wb") as f:
                for data in output_volume.read_file(output_filename):
                    f.write(data)
            downloaded += 1
        except Exception:
            pass

        # Download stderr log
        stderr_filename = f"logs_{run_name}/level_{level}_problem_{problem_id}_sample_0_stderr.txt"
        try:
            filepath = os.path.join(logs_dir, f"level_{level}_problem_{problem_id}_sample_0_stderr.txt")
            with open(filepath, "wb") as f:
                for data in output_volume.read_file(stderr_filename):
                    f.write(data)
            downloaded += 1
        except Exception:
            pass

    print(f"Downloaded {downloaded} log files to {logs_dir}")
    return downloaded

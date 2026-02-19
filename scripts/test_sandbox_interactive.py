#!/usr/bin/env python3
"""
Interactive Sandbox Shell for KernelBench Agent Image.

Creates a Modal Sandbox with the agent_image and drops you into an
interactive bash session. Useful for:
  - Verifying claude CLI is installed and works
  - Testing AWS credentials
  - Manually running claude prompts
  - Debugging the sandbox environment

Usage:
    cd KernelBench
    uv run python scripts/test_sandbox_interactive.py

    # With options
    uv run python scripts/test_sandbox_interactive.py gpu=A10G timeout=1800
"""

import os
import sys
import select
import threading

import modal
import pydra
from pydra import Config

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from batch_run_modal.config import GPU_ARCH_MAPPING
from batch_run_modal.images import agent_image, eval_image


class SandboxConfig(Config):
    def __init__(self):
        self.gpu = "A10G"
        self.timeout = 3600
        self.anthropic_model = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.anthropic_small_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pydra.main(base=SandboxConfig)
def main(config: SandboxConfig):
    print("=========================================")
    print("KernelBench Interactive Sandbox")
    print("=========================================")
    print(f"GPU: {config.gpu}")
    print(f"Timeout: {config.timeout}s")
    print("=========================================")

    # AWS credentials are now included in the agent_image
    print("[INFO] AWS credentials are included in the agent_image")

    env = {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_REGION": "us-east-1",
        "AWS_PROFILE": "bedrock",
        "ANTHROPIC_MODEL": config.anthropic_model,
        "ANTHROPIC_SMALL_FAST_MODEL": config.anthropic_small_model,
    }

    app = modal.App.lookup("kb-sandbox-interactive", create_if_missing=True)

    print("\nCreating sandbox (this may take a while on first run for image build)...")

    sb = modal.Sandbox.create(
        app=app,
        image=agent_image,
        gpu=config.gpu,
        env=env,
        timeout=config.timeout,
        workdir="/root",
        cpu=4.0,
        memory=32768,
    )
    print(f"Sandbox created: {sb.object_id}")
    process = sb.exec("python", "-c", "import sys; print(sys.executable)", timeout=3)
    print(process.stdout.read())
    def run_cmd(label, *args, pty=False, timeout=60):
        print(f"\n{'='*50}")
        print(f"[TEST] {label}")
        print('='*50)
        try:
            proc = sb.exec(*args, pty=pty, timeout=timeout)
            stdout_lines = []
            stderr_lines = []

            for line in proc.stdout:
                print(line, end="")
                stdout_lines.append(line)
            for line in proc.stderr:
                print(line, end="", file=sys.stderr)
                stderr_lines.append(line)

            proc.wait()
            print(f"[exit {proc.returncode}]")

            if proc.returncode != 0:
                print(f"[DEBUG] stdout lines: {len(stdout_lines)}")
                print(f"[DEBUG] stderr lines: {len(stderr_lines)}")
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)

    try:
        # 1. 检查环境变量
        # run_cmd("Check AWS env", "bash", "-c", "env | grep -E '(AWS|ANTHROPIC|CLAUDE)' | sort")

        # run_cmd(
        #     'claude -p "hello" (with PTY)',
        #     "claude", "-p", "你好", "--allowedTools", "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell",
        #     pty=True, # must have pty=True for claude CLI to work properly
        #     timeout=500
        # )
        # run_cmd(
        #     'claude -p "hello" (with PTY)',
        #     "claude", "-p", "写一个cuda程序，并且用ncu basic测试，告诉我结果", "--allowedTools", "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell",
        #     pty=True,
        #     timeout=500
        # )

        run_cmd(
            'claude via bash -c (with PTY)',
            "bash", "-c", 'claude -p "你好" --allowedTools "Read,Edit,Bash,WebFetch,WebSearch,Write,Glob,Grep,KillShell" --output-format stream-json --verbose --include-partial-messages | jq -rj \'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text\'',
            pty=True,
            timeout=30
        )


    finally:
        print("\nTerminating sandbox...")
        sb.terminate()
        print("Done.")


if __name__ == "__main__":
    main()

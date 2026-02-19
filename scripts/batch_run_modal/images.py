"""
Modal Image definitions for KernelBench.

Two images are built on top of each other:
    eval_image  – CUDA + Python + KernelBench deps (for evaluation)
    agent_image – eval_image + Claude CLI + utilities  (for agent generation)
"""

import os

import modal

from .config import REPO_TOP_DIR

# Local source directories to mount into images
SRC_DIR = os.path.join(REPO_TOP_DIR, "src")
KERNELBENCH_DIR = os.path.join(REPO_TOP_DIR, "KernelBench")
SCRIPTS_DIR = os.path.join(REPO_TOP_DIR, "scripts")
EXAMPLES_DIR = os.path.join(REPO_TOP_DIR, "runs/example")


# ---------------------------------------------------------------------------
# Base evaluation image — shared foundation for both eval and agent
# NOTE: add_local_dir must be LAST — no build steps allowed after it.
# ---------------------------------------------------------------------------
_eval_base = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .uv_sync(uv_project_dir=REPO_TOP_DIR, extras=["gpu"])
    .env({"PYTHONPATH": "/root/src:/root"})
)

eval_image = (
    _eval_base
    .add_local_dir(SRC_DIR, remote_path="/root/src")
    .add_local_dir(KERNELBENCH_DIR, remote_path="/root/KernelBench")
)

# ---------------------------------------------------------------------------
# Agent image = base + Claude CLI + jq/curl/vim, then add_local_dir last
# ---------------------------------------------------------------------------

# Check if AWS credentials directory exists
_aws_dir = os.path.expanduser("~/.aws")
_has_aws = os.path.exists(_aws_dir)

_agent_with_tools = (
    _eval_base
    .apt_install("jq", "curl", "vim", "less")
    .run_commands(
        # Install Claude CLI (installs to ~/.local/bin/)
        "curl -fsSL https://claude.ai/install.sh | bash",
    )
    .env({"PATH": "/root/.local/bin:$PATH"})
    .add_local_dir(SRC_DIR, remote_path="/root/src")
    .add_local_dir(KERNELBENCH_DIR, remote_path="/root/KernelBench")
    .add_local_dir(SCRIPTS_DIR, remote_path="/root/scripts")
    .add_local_dir(EXAMPLES_DIR, remote_path="/root/runs/example")
)

# Add AWS credentials if available
if _has_aws:
    agent_image = _agent_with_tools.add_local_dir(_aws_dir, remote_path="/root/.aws")
    print(f"[INFO] AWS credentials will be included in agent_image from {_aws_dir}")
else:
    agent_image = _agent_with_tools
    print(f"[WARN] AWS credentials directory not found at {_aws_dir}, skipping")

"""
Configuration classes and constants for KernelBench Modal batch runs.
"""

import os
from datetime import datetime

from pydra import Config, REQUIRED

# Root of the KernelBench repository
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPU architecture mapping used by CUDA toolchain and eval
GPU_ARCH_MAPPING = {
    "A10G": "Ampere",
    "A100": "Ampere",
    "A100-80GB": "Ampere",
    "L40S": "Ada",
    "L4": "Ada",
    "H100": "Hopper",
    "T4": "Turing",
}


class BatchRunConfig(Config):
    """All tunables for a batch run."""

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
        self.sandbox_timeout = 2400             # 40 minutes

        # Evaluation settings
        self.eval_timeout = 300                 # Eval subprocess timeout
        self.baseline = "baseline_time_torch"   # Baseline name for analysis

        # Resume mode: skip already completed problems
        self.resume = False
        # Enable NVIDIA Nsight Compute profiling
        self.enable_ncu = True

        # AWS profile name (used with `aws sso login --profile=xxx`)
        self.aws_profile = "bedrock"

        # Anthropic model config
        # self.anthropic_model = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        # self.anthropic_model = "global.anthropic.claude-opus-4-6-v1"
        self.anthropic_model = "global.anthropic.claude-sonnet-4-6"
        self.anthropic_small_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

        # Whether to run evaluation + analysis after generation
        self.run_eval = True

    def __repr__(self):
        return f"BatchRunConfig({self.to_dict()})"

    def coerce_booleans(self):
        """Convert string booleans from CLI to actual bools."""
        for attr in ("resume", "enable_ncu", "run_eval"):
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, val.lower() in ("true", "1", "yes"))

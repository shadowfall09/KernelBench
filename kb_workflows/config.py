from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .common import GPU_ARCH, ROOT, safe_name


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"1", "true", "yes"}:
        return True
    if str(value).lower() in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


@dataclass
class RunConfig:
    level: int = 1
    start: int = 1
    end: int | None = None
    run_name: str = ""
    runs_dir: str = str(ROOT / "runs")
    stage: str = "all"
    generation: str = "docker"
    evaluation: str = "local"
    gpus: str = "0"
    gpu: str = "H100"
    gpu_arch: str = ""
    hardware: str = ""
    parallel: int = 4
    num_samples: int = 1
    pass_at_k: str = "1,5"
    timeout: int = 300
    agent_timeout: int = 2400
    optimization_rounds: int = 5
    resume: bool = False
    enable_ncu: bool = True
    precision: str = "fp32"
    fp32_tolerance: float = float(os.environ.get("KB_FP32_TOL", "1e-3"))
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    hidden_tests: bool = True
    baseline: str = ""
    record_baseline: bool = False
    compile_backend: str = ""
    compile_mode: str = "default"
    profile: str = ""
    model: str = ""
    small_model: str = ""
    bedrock: bool = False
    docker_image: str = "kernelbench-verified:agent"
    build_image: bool = True
    queue_url: str = os.environ.get("KB_QUEUE_BASE_URL", "http://127.0.0.1:8000")
    queue_submit_path: str = "/v1/verify"
    queue_result_path: str = "/v1/verify/{request_id}"
    queue_max_inflight: int = 8
    queue_poll_interval: float = 0.5
    queue_http_timeout: float = 30.0
    queue_wait_timeout: int = 3600
    modal_secret: str = ""
    dry_run: bool = False

    def validate(self):
        self.run_name = safe_name(self.run_name or f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
        self.runs_dir = str(Path(self.runs_dir).expanduser().resolve())
        if self.level not in (1, 2, 3) or self.start < 1 or (self.end is not None and self.end < self.start):
            raise ValueError("Verified supports levels 1–3; start/end must form a valid inclusive range")
        for name in (
            "parallel",
            "num_samples",
            "timeout",
            "agent_timeout",
            "optimization_rounds",
            "num_correct_trials",
            "num_perf_trials",
            "queue_max_inflight",
            "queue_wait_timeout",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.queue_poll_interval <= 0 or self.queue_http_timeout <= 0:
            raise ValueError("Queue polling interval and HTTP timeout must be positive")
        if not math.isfinite(self.fp32_tolerance) or self.fp32_tolerance <= 0:
            raise ValueError("FP32 tolerance must be finite and positive")
        devices = self.gpus.split(",")
        if any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices):
            raise ValueError("--gpus must contain distinct numeric device IDs")
        self.gpu_arch = self.gpu_arch or GPU_ARCH.get(self.hardware, GPU_ARCH.get(self.gpu, "Hopper"))
        self.hardware = self.hardware or (self.gpu if self.evaluation == "modal" else "")
        suffix = "tf32" if self.precision == "fp32" else self.precision
        prefix = (
            f"baseline_time_torch_compile_{self.compile_backend}_{self.compile_mode}"
            if self.compile_backend
            else "baseline_time_torch"
        )
        self.baseline = self.baseline or f"{prefix}_{suffix}"
        safe_name(self.baseline)
        if self.hardware and Path(self.hardware).name != self.hardware:
            raise ValueError("Hardware must be a directory label, not a path")
        if self.profile:
            self.bedrock = True
        self.model = self.model or os.environ.get("ANTHROPIC_MODEL", "")
        self.small_model = self.small_model or os.environ.get("ANTHROPIC_SMALL_FAST_MODEL", "")
        self.bedrock = self.bedrock or os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1"
        if any(int(k) <= 0 for k in self.pass_at_k.split(",")):
            raise ValueError("pass-at-k values must be positive")
        return self

    @property
    def run_dir(self):
        return Path(self.runs_dir) / self.run_name

    def to_dict(self):
        return asdict(self)


def parser():
    p = argparse.ArgumentParser(
        description="KernelBench-Verified: generate, evaluate, and analyze with one configuration"
    )
    for name in (
        "level",
        "start",
        "end",
        "parallel",
        "num_samples",
        "timeout",
        "agent_timeout",
        "optimization_rounds",
        "num_correct_trials",
        "num_perf_trials",
        "queue_max_inflight",
        "queue_wait_timeout",
    ):
        aliases = [f"--{name.replace('_', '-')}"]
        if name == "parallel":
            aliases.append("--num-parallel")
        p.add_argument(*aliases, dest=name, type=int, default=getattr(RunConfig(), name))
    choices = {
        "stage": ("all", "generate", "evaluate", "analyze", "baseline"),
        "generation": ("local", "docker", "modal"),
        "evaluation": ("local", "docker", "modal", "queue"),
        "precision": ("fp32", "fp16", "bf16"),
    }
    for name in (
        "run_name",
        "runs_dir",
        "stage",
        "generation",
        "evaluation",
        "gpus",
        "gpu",
        "gpu_arch",
        "hardware",
        "precision",
        "baseline",
        "compile_backend",
        "compile_mode",
        "profile",
        "model",
        "small_model",
        "docker_image",
        "queue_url",
        "queue_submit_path",
        "queue_result_path",
        "modal_secret",
        "pass_at_k",
    ):
        p.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            default=getattr(RunConfig(), name),
            **({"choices": choices[name]} if name in choices else {}),
        )
    for name in ("resume", "enable_ncu", "hidden_tests", "record_baseline", "bedrock", "build_image", "dry_run"):
        flag = name.replace("_", "-")
        p.add_argument(f"--{flag}", dest=name, type=boolean, nargs="?", const=True, default=getattr(RunConfig(), name))
        p.add_argument(f"--no-{flag}", dest=name, action="store_false")
    for name in ("queue_poll_interval", "queue_http_timeout", "fp32_tolerance"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float, default=getattr(RunConfig(), name))
    return p


def parse_config(argv=None):
    return RunConfig(**vars(parser().parse_args(argv))).validate()

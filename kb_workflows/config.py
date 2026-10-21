from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, safe_name
from .gpu import cuda_architectures, gpu_options, gpu_profiles, read_toml


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
    gpu: str = ""
    gpu_arch: str = ""
    hardware: str = ""
    parallel: int = 4
    num_samples: int = 1
    pass_at_k: str = "1,5"
    timeout: int = 300
    agent_timeout: int = 2400
    optimization_rounds: int = 5
    generation_mode: str = "feedback"
    agent_backend: str = "claude-code"
    agent_cost_limit: float = 0.0
    agent_profile: bool = True
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
    baseline_suite: bool = True
    cold_calls: int = 10000
    problem_specs: str = ""
    score_gate_required: bool = True
    profile: str = ""
    model: str = ""
    small_model: str = ""
    bedrock: bool = False
    docker_image: str = "kernelbench-pro:agent"
    build_image: bool = True
    queue_url: str = os.environ.get("KB_QUEUE_BASE_URL", "http://127.0.0.1:8000")
    queue_submit_path: str = "/v1/verify"
    queue_result_path: str = "/v1/verify/{request_id}"
    queue_max_inflight: int = 8
    queue_poll_interval: float = 0.5
    queue_http_timeout: float = 30.0
    queue_wait_timeout: int = 3600
    modal_secret: str = ""
    harbor_command: str = "harbor"
    harbor_agent: str = "claude-code"
    harbor_environment: str = "docker"
    harbor_reward: str = "correctness"
    harbor_memory_mb: int = 65536
    dry_run: bool = False

    def validate(self):
        self.run_name = safe_name(self.run_name or f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
        self.runs_dir = str(Path(self.runs_dir).expanduser().resolve())
        if self.level not in (1, 2, 3) or self.start < 1 or (self.end is not None and self.end < self.start):
            raise ValueError("KernelBench Pro batch workflows support levels 1–3; start/end must form a valid inclusive range")
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
            "cold_calls",
            "harbor_memory_mb",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.queue_poll_interval <= 0 or self.queue_http_timeout <= 0:
            raise ValueError("Queue polling interval and HTTP timeout must be positive")
        if not math.isfinite(self.fp32_tolerance) or self.fp32_tolerance <= 0:
            raise ValueError("FP32 tolerance must be finite and positive")
        if not math.isfinite(self.agent_cost_limit) or self.agent_cost_limit < 0:
            raise ValueError("agent-cost-limit must be finite and nonnegative; 0 disables the estimated-cost cap")
        if self.generation_mode not in {"one-shot", "feedback", "external"}:
            raise ValueError("Invalid generation mode")
        if self.agent_backend not in {"claude-code", "litellm"}:
            raise ValueError("Invalid agent backend")
        if self.generation == "harbor":
            if self.generation_mode == "one-shot":
                raise ValueError("Harbor agents cannot certify one-shot; use native generation or external mode")
            self.generation_mode = "external"
        if self.generation_mode == "external" and self.generation != "harbor" and self.stage in {"all", "generate"}:
            raise ValueError("external generation mode is reserved for Harbor")
        if self.problem_specs:
            self.problem_specs = str(Path(self.problem_specs).expanduser().resolve())
            if not Path(self.problem_specs).is_file():
                raise ValueError(f"Problem specs file not found: {self.problem_specs}")
        if self.baseline_suite and self.compile_backend:
            raise ValueError("Use --baseline-suite or --compile-backend, not both")
        if self.harbor_environment not in {"docker", "modal"} or self.harbor_reward not in {"correctness", "bounded_gain"}:
            raise ValueError("Invalid Harbor environment or reward mode")
        if not self.harbor_command.strip() or not self.harbor_agent.strip():
            raise ValueError("Harbor command and agent must not be empty")
        if (self.generation == "harbor" or self.stage == "export-harbor") and not self.hidden_tests:
            raise ValueError("Harbor requires hidden correctness tests")
        devices = self.gpus.split(",")
        if any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices):
            raise ValueError("--gpus must contain distinct numeric device IDs")
        profiles = gpu_profiles()
        target = self.hardware or self.gpu
        preset = profiles.get(target, {})
        if target and not self.gpu_arch and not preset:
            raise ValueError(f"Unknown GPU {target!r}; set gpu.hardware and gpu.architecture in --config, or --gpu-arch")
        self.gpu_arch = self.gpu_arch or preset.get("gpu_arch", "")
        if self.gpu_arch:
            self.gpu_arch = cuda_architectures(self.gpu_arch)
        self.hardware = self.hardware or preset.get("hardware", self.gpu)
        self.gpu = self.gpu or preset.get("gpu", "")
        suffix = "tf32" if self.precision == "fp32" and not self.baseline_suite else self.precision
        prefix = (
            f"baseline_time_torch_compile_{self.compile_backend}_{self.compile_mode}"
            if self.compile_backend
            else "baseline_time_torch"
        )
        if self.baseline_suite:
            prefix = "baseline_time_suite"
        self.baseline = self.baseline or f"{prefix}_{suffix}_wall"
        safe_name(self.baseline)
        if self.hardware and Path(self.hardware).name != self.hardware:
            raise ValueError("Hardware must be a directory label, not a path")
        if self.profile:
            self.bedrock = True
        self.model = self.model or os.environ.get("ANTHROPIC_MODEL", "")
        if self.agent_backend == "litellm" and not self.model and self.generation != "harbor" and self.stage in {"all", "generate"}:
            raise ValueError("LiteLLM requires an explicit provider/model via --model or run.model")
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

    def require_gpu_target(self):
        if not self.hardware or not self.gpu_arch:
            raise ValueError("Select a GPU with --config / --gpu-profile, or specify --hardware and --gpu-arch")
        cloud = (self.evaluation == "modal" and self.stage in {"all", "evaluate", "baseline"}) or (
            self.generation == "modal" and self.stage in {"all", "generate"}
        ) or (self.harbor_environment == "modal" and (
            self.stage == "export-harbor" or (self.generation == "harbor" and self.stage in {"all", "generate"})
        ))
        if cloud and not self.gpu:
            raise ValueError("This backend needs gpu.modal_type in --config (or --gpu); no cloud GPU is chosen implicitly")


def parser():
    p = argparse.ArgumentParser(
        description="KernelBench Pro: generate, evaluate, and analyze with one configuration"
    )
    p.add_argument("--config", type=Path, help="TOML file with [run] and [gpu] sections; CLI flags override it")
    p.add_argument("--gpu-profile", help="GPU preset from kb_workflows/gpu_profiles.toml")
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
        "cold_calls",
        "harbor_memory_mb",
    ):
        aliases = [f"--{name.replace('_', '-')}"]
        if name == "parallel":
            aliases.append("--num-parallel")
        p.add_argument(*aliases, dest=name, type=int, default=argparse.SUPPRESS)
    choices = {
        "stage": ("all", "generate", "evaluate", "analyze", "baseline", "export-harbor"),
        "generation": ("local", "docker", "modal", "harbor"),
        "evaluation": ("local", "docker", "modal", "queue"),
        "precision": ("fp32", "fp16", "bf16"),
        "harbor_environment": ("docker", "modal"),
        "harbor_reward": ("correctness", "bounded_gain"),
        "generation_mode": ("one-shot", "feedback", "external"),
        "agent_backend": ("claude-code", "litellm"),
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
        "problem_specs",
        "harbor_command",
        "harbor_agent",
        "harbor_environment",
        "harbor_reward",
        "generation_mode",
        "agent_backend",
    ):
        p.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            default=argparse.SUPPRESS,
            **({"choices": choices[name]} if name in choices else {}),
        )
    for name in ("resume", "enable_ncu", "hidden_tests", "record_baseline", "bedrock", "build_image", "dry_run", "baseline_suite", "score_gate_required", "agent_profile"):
        flag = name.replace("_", "-")
        p.add_argument(f"--{flag}", dest=name, type=boolean, nargs="?", const=True, default=argparse.SUPPRESS)
        p.add_argument(f"--no-{flag}", dest=name, action="store_false", default=argparse.SUPPRESS)
    for name in ("queue_poll_interval", "queue_http_timeout", "fp32_tolerance", "agent_cost_limit"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float, default=argparse.SUPPRESS)
    return p


def file_options(path, argument_parser):
    path = Path(path).expanduser().resolve()
    document = read_toml(path)
    if unknown := set(document) - {"run", "gpu"}:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    run = document.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("[run] must be a table")
    defaults = RunConfig().to_dict()
    gpu_fields = {"hardware", "gpu", "gpu_arch", "gpus"}
    actions = {action.dest: action for action in argument_parser._actions}
    options = {}
    for key, value in run.items():
        if key in gpu_fields:
            raise ValueError(f"Put GPU settings in [gpu], not run.{key}")
        if key not in defaults:
            raise ValueError(f"Unknown [run] option: {key}")
        expected = type(defaults[key]) if defaults[key] is not None else int
        if expected is float and type(value) is int:
            value = float(value)
        if type(value) is not expected:
            raise ValueError(f"run.{key} must be {expected.__name__}, got {type(value).__name__}")
        if actions[key].choices and value not in actions[key].choices:
            raise ValueError(f"Invalid run.{key}: {value!r}; choose from {actions[key].choices}")
        if key in {"runs_dir", "problem_specs"} and value:
            candidate = Path(value).expanduser()
            value = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
        options[key] = value
    options.update(gpu_options(document.get("gpu", {})))
    return options


def parse_config(argv=None):
    argument_parser = parser()
    arguments = vars(argument_parser.parse_args(argv))
    path = arguments.pop("config")
    profile = arguments.pop("gpu_profile")
    options = file_options(path, argument_parser) if path else {}
    if profile is not None:
        options.update(gpu_options({"profile": profile}))
    # Changing the target through legacy CLI flags must not retain the file's
    # architecture or cloud type for a different GPU.
    if "hardware" in arguments or "gpu" in arguments:
        options.update(hardware="", gpu="", gpu_arch="")
    options.update(arguments)
    return RunConfig(**options).validate()

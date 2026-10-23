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
    suite: str = "full"
    suite_role: str = "scored"
    task_profile: str = "standard"
    task_track: str = "all"
    task_memory_mb: int = 512
    task_dimensions: str = ""
    device_backend: str = "cuda"
    device_python: str = ""
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
    agent_token_limit: int = 0
    agent_output_token_limit: int = 0
    agent_max_output_tokens: int = 8192
    agent_device_seconds: float = 0.0
    agent_temperature: float = 0.0
    local_model_device: str = "cpu"
    agent_profile: bool = True
    resume: bool = False
    enable_ncu: bool = True
    sanitizer_tools: str = "none"
    sanitizer_command: str = "compute-sanitizer"
    sanitizer_timeout: int = 600
    determinism_checks: bool = False
    state_validation: bool = False
    numerical_validation: bool = False
    input_calibration: str = ""
    input_family_memory_mb: int = 256
    oracle_memory_mb: int = 4096
    determinism_repeats: int = 5
    determinism_policy_file: str = ""
    isolation_mode: str = "standard"
    isolation_cpus: int = 4
    isolation_memory_mb: int = 65536
    isolation_pids: int = 512
    isolation_tmp_mb: int = 16384
    isolation_report_mb: int = 64
    regrade_manifest: str = ""
    precision: str = "fp32"
    fp32_tolerance: float = float(os.environ.get("KB_FP32_TOL", "1e-3"))
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    hidden_tests: bool = True
    strict_validation: bool = False
    evidence_level: str = "current"
    baseline: str = ""
    record_baseline: bool = False
    compile_backend: str = ""
    compile_mode: str = "default"
    baseline_suite: bool = True
    baseline_portfolio: str = "standard"
    paired_timing: bool = False
    paired_trials: int = 30
    final_repeats: int = 0
    cold_calls: int = 10000
    problem_specs: str = ""
    score_mode: str = "bounded_gain"
    device_calibration: str = ""
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
    harbor_agent_hosts: str = ""
    dry_run: bool = False

    def validate(self):
        self.run_name = safe_name(self.run_name or f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
        self.runs_dir = str(Path(self.runs_dir).expanduser().resolve())
        if self.level not in (1, 2, 3, 5) or self.start < 1 or (self.end is not None and self.end < self.start):
            raise ValueError("Levels must be 1–3 or 5; start/end must form a valid inclusive range")
        if (self.level == 5) != (self.suite == "production41"):
            raise ValueError("Level 5 requires --suite production41, which only contains level 5")
        if self.suite == "production41" and self.suite_role != "scored":
            raise ValueError("production41 contains only the scored role")
        if self.task_profile not in {"small", "standard", "large"}:
            raise ValueError("Unknown production task profile")
        if self.task_dimensions:
            if self.suite != "production41":
                raise ValueError("Explicit task dimensions require production41")
            from .production import read_dimensions

            self.task_dimensions = str(Path(self.task_dimensions).expanduser().resolve())
            read_dimensions(self.task_dimensions)
        if self.task_track not in {"all", "design", "numerical", "implementation", "backward"}:
            raise ValueError("Unknown production task track")
        if self.suite != "production41" and self.task_track != "all":
            raise ValueError("--task-track requires production41")
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
            "task_memory_mb",
            "sanitizer_timeout",
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
        if self.agent_backend not in {"claude-code", "litellm", "local-transformers"}:
            raise ValueError("Invalid agent backend")
        for name in ("agent_token_limit", "agent_output_token_limit", "agent_max_output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (1 if name == "agent_max_output_tokens" else 0):
                raise ValueError(f"{name} must be an integer in its nonnegative/positive range")
        for name in ("agent_device_seconds", "agent_temperature"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        from .resource_budgets import active

        if active(self) and self.generation not in {"local", "docker"}:
            raise ValueError("Resource reservations require native local/docker generation; remote/Harbor enforcement is unavailable")
        if self.agent_token_limit and self.agent_backend != "local-transformers":
            raise ValueError("Exact total-token caps require local-transformers; provider token estimates cannot certify this cap")
        if self.agent_output_token_limit and self.agent_backend not in {"local-transformers", "litellm"}:
            raise ValueError("Output-token reservations require local-transformers or a supported single-call LiteLLM provider")
        if self.agent_device_seconds and self.evaluation not in {"local", "docker"}:
            raise ValueError("Device lease reservations require local/docker evaluation with parent-owned timing")
        if self.agent_backend == "local-transformers":
            if self.generation != "local":
                raise ValueError("Local checkpoints currently require generation=local; container checkpoint mounting is unavailable")
            if self.agent_cost_limit:
                raise ValueError("Local model billing is not measured; use token/device reservations rather than a dollar cap")
            if not isinstance(self.local_model_device, str) or not self.local_model_device:
                raise ValueError("local_model_device must name an explicit Torch device")
        if self.generation == "harbor":
            if self.generation_mode == "one-shot":
                raise ValueError("Harbor agents cannot certify one-shot; use native generation or external mode")
            self.generation_mode = "external"
        if self.generation_mode == "external" and self.generation != "harbor" and self.stage in {"all", "generate"}:
            raise ValueError("external generation mode is reserved for Harbor")
        from .suites import ROLES, SUITES

        if self.suite not in SUITES or self.suite_role not in ROLES:
            raise ValueError("Invalid suite or suite role")
        if self.suite == "full" and self.suite_role != "scored":
            raise ValueError("--suite-role requires --suite kernelbench105")
        if self.evidence_level not in {"current", "L0", "L1", "L2"}:
            raise ValueError("Unknown evidence level")
        if self.evidence_level != "current":
            self.strict_validation = True
        from .tool_checks import select_tools

        tools = select_tools(self.sanitizer_tools)
        self.sanitizer_tools = ",".join(tools) if tools else "none"
        if not isinstance(self.sanitizer_command, str) or not self.sanitizer_command.strip():
            raise ValueError("sanitizer-command must name one executable")
        if tools and self.device_backend != "cuda":
            raise ValueError("Compute Sanitizer requires CUDA; Neuron has no equivalent tool suite")
        if type(self.determinism_checks) is not bool or type(self.determinism_repeats) is not int or not 3 <= self.determinism_repeats <= 100:
            raise ValueError("Determinism checks require an explicit boolean and 3 to 100 repeats")
        if type(self.state_validation) is not bool:
            raise ValueError("State validation requires an explicit boolean")
        if self.input_calibration:
            from .input_calibration import read_bundle

            if not isinstance(self.input_calibration, str) or self.level != 1:
                raise ValueError("Input calibration currently requires a file and original level 1")
            self.input_calibration = str(Path(self.input_calibration).expanduser().resolve())
            read_bundle(self.input_calibration)
            self.numerical_validation = True
        if type(self.numerical_validation) is not bool or (self.numerical_validation and self.level != 1):
            raise ValueError("Independent numerical validation currently requires an explicit boolean and level 1")
        if type(self.input_family_memory_mb) is not int or not 1 <= self.input_family_memory_mb <= 1048576:
            raise ValueError("Input-family memory envelope must be an integer from 1 to 1048576 MiB")
        if type(self.oracle_memory_mb) is not int or not 1 <= self.oracle_memory_mb <= 1048576:
            raise ValueError("Oracle host-memory envelope must be 1..1048576 MiB")
        if self.determinism_policy_file:
            from .determinism import read_policies

            self.determinism_policy_file = str(Path(self.determinism_policy_file).expanduser().resolve())
            read_policies(self.determinism_policy_file)
        from .isolation import allowed_hosts, options, policy

        policy(options(self))
        hosts = allowed_hosts(self.harbor_agent_hosts)
        self.harbor_agent_hosts = ",".join(hosts)
        if self.isolation_mode == "restricted":
            if self.evaluation != "docker" or self.device_backend != "cuda":
                raise ValueError("Restricted verification requires evaluation=docker and CUDA")
            if self.generation == "harbor" and self.harbor_environment != "docker":
                raise ValueError("Restricted Harbor verification requires its Docker provider")
            if self.generation == "harbor" and self.stage in {"all", "generate"} and not hosts:
                raise ValueError("Restricted Harbor agent execution requires explicit --harbor-agent-hosts")
            if self.generation in {"docker", "modal"} and self.stage in {"all", "generate"}:
                raise ValueError("Restricted verification requires a separate verifier; use local or Harbor generation")
        if self.regrade_manifest:
            self.regrade_manifest = str(Path(self.regrade_manifest).expanduser().resolve())
            if self.stage in {"all", "generate"}:
                raise ValueError("Artifact regrading cannot start generation")
        if self.problem_specs:
            self.problem_specs = str(Path(self.problem_specs).expanduser().resolve())
            if not Path(self.problem_specs).is_file():
                raise ValueError(f"Problem specs file not found: {self.problem_specs}")
        if self.score_mode not in {"bounded_gain", "headroom"}:
            raise ValueError("Unknown score mode")
        if self.device_calibration:
            self.device_calibration = str(Path(self.device_calibration).expanduser().resolve())
            if not Path(self.device_calibration).is_file():
                raise ValueError("Device calibration file not found")
        if self.score_mode == "headroom" and not (self.problem_specs and self.device_calibration):
            raise ValueError("Headroom scoring requires reviewed problem specs and a measured device calibration")
        if self.baseline_portfolio not in {"standard", "extended"}:
            raise ValueError("Unknown baseline portfolio")
        if type(self.final_repeats) is not int or self.final_repeats < 0:
            raise ValueError("final_repeats must be nonnegative")
        if type(self.paired_trials) is not int or self.paired_trials < 10:
            raise ValueError("paired_trials must be at least 10")
        if self.paired_timing and not self.baseline_suite:
            raise ValueError("Paired timing requires the validated baseline portfolio")
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
        if self.device_backend not in {"cuda", "neuron"}:
            raise ValueError("Unknown device backend")
        if self.device_python:
            self.device_python = str(Path(self.device_python).expanduser().resolve())
        profiles = gpu_profiles()
        target = self.hardware or self.gpu
        preset = profiles.get(target, {})
        if self.device_backend == "neuron":
            from .devices import NEURON_TARGETS

            if target not in NEURON_TARGETS or self.gpu_arch not in {"", NEURON_TARGETS.get(target)}:
                raise ValueError("Neuron requires Trainium2/trn2 or Trainium3/trn3")
            self.gpu_arch = NEURON_TARGETS[target]
            if self.gpu or self.evaluation not in {"local", "queue"} or self.generation != "local" or self.stage == "export-harbor":
                raise ValueError("Neuron uses local/queue verification and native generation; CUDA Docker/Modal/Harbor images are not Neuron environments")
            if not self.baseline_suite or self.baseline_portfolio != "standard" or self.compile_backend or self.enable_ncu:
                raise ValueError("Neuron requires its standard eager/compile portfolio; CUDA compilation and Nsight options do not apply")
        elif target and not self.gpu_arch and not preset:
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
            prefix = "baseline_time_suite" if self.baseline_portfolio == "standard" else "baseline_time_suite_extended"
        self.baseline = self.baseline or f"{prefix}_{suffix}_wall"
        safe_name(self.baseline)
        if self.hardware and Path(self.hardware).name != self.hardware:
            raise ValueError("Hardware must be a directory label, not a path")
        if self.profile:
            self.bedrock = True
        self.model = self.model or (os.environ.get("ANTHROPIC_MODEL", "") if self.agent_backend != "local-transformers" else "")
        if self.agent_backend == "local-transformers":
            self.model = str(Path(self.model).expanduser().resolve()) if self.model else ""
            if self.stage in {"all", "generate"} and (not self.model or not Path(self.model).is_dir()):
                raise ValueError("local-transformers requires an existing local checkpoint directory")
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
        "paired_trials",
        "final_repeats",
        "task_memory_mb",
        "agent_token_limit",
        "agent_output_token_limit",
        "agent_max_output_tokens",
        "sanitizer_timeout",
        "oracle_memory_mb",
        "input_family_memory_mb",
        "determinism_repeats",
        "isolation_cpus",
        "isolation_memory_mb",
        "isolation_pids",
        "isolation_tmp_mb",
        "isolation_report_mb",
    ):
        aliases = [f"--{name.replace('_', '-')}"]
        if name == "parallel":
            aliases.append("--num-parallel")
        p.add_argument(*aliases, dest=name, type=int, default=argparse.SUPPRESS)
    choices = {
        "device_backend": ("cuda", "neuron"),
        "score_mode": ("bounded_gain", "headroom"),
        "baseline_portfolio": ("standard", "extended"),
        "evidence_level": ("current", "L0", "L1", "L2"),
        "suite": ("full", "kernelbench105", "production41"),
        "task_profile": ("small", "standard", "large"),
        "task_track": ("all", "design", "numerical", "implementation", "backward"),
        "suite_role": ("scored", "probe", "extended"),
        "stage": ("all", "generate", "evaluate", "analyze", "baseline", "export-harbor"),
        "generation": ("local", "docker", "modal", "harbor"),
        "evaluation": ("local", "docker", "modal", "queue"),
        "precision": ("fp32", "fp16", "bf16"),
        "harbor_environment": ("docker", "modal"),
        "harbor_reward": ("correctness", "bounded_gain"),
        "generation_mode": ("one-shot", "feedback", "external"),
        "agent_backend": ("claude-code", "litellm", "local-transformers"),
    }
    for name in (
        "score_mode",
        "device_calibration",
        "input_calibration",
        "baseline_portfolio",
        "evidence_level",
        "suite",
        "suite_role",
        "task_profile",
        "task_dimensions",
        "device_backend",
        "device_python",
        "sanitizer_tools",
        "sanitizer_command",
        "determinism_policy_file",
        "isolation_mode",
        "regrade_manifest",
        "task_track",
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
        "harbor_agent_hosts",
        "harbor_agent",
        "harbor_environment",
        "harbor_reward",
        "generation_mode",
        "agent_backend",
        "local_model_device",
    ):
        p.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            default=argparse.SUPPRESS,
            **({"choices": choices[name]} if name in choices else {}),
        )
    for name in ("paired_timing", "resume", "enable_ncu", "hidden_tests", "strict_validation", "record_baseline", "bedrock", "build_image", "dry_run", "baseline_suite", "score_gate_required", "agent_profile", "determinism_checks", "state_validation", "numerical_validation"):
        flag = name.replace("_", "-")
        p.add_argument(f"--{flag}", dest=name, type=boolean, nargs="?", const=True, default=argparse.SUPPRESS)
        p.add_argument(f"--no-{flag}", dest=name, action="store_false", default=argparse.SUPPRESS)
    for name in ("queue_poll_interval", "queue_http_timeout", "fp32_tolerance", "agent_cost_limit", "agent_device_seconds", "agent_temperature"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float, default=argparse.SUPPRESS)
    return p


def file_options(path, argument_parser):
    path = Path(path).expanduser().resolve()
    document = read_toml(path)
    if unknown := set(document) - {"run", "gpu", "device"}:
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
        if key in {"runs_dir", "problem_specs", "device_calibration", "input_calibration", "task_dimensions", "determinism_policy_file", "regrade_manifest"} and value:
            candidate = Path(value).expanduser()
            value = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
        options[key] = value
    if document.get("gpu") and document.get("device"):
        raise ValueError("Use either [gpu] or [device] for one device target")
    if "device" in document:
        from .devices import device_options

        options.update(device_options(document["device"], directory=path.parent))
    else:
        options.update(gpu_options(document.get("gpu", {})))
    if options.get("device_backend") == "neuron":
        options.setdefault("generation", "local")
        options.setdefault("enable_ncu", False)
    if options.get("agent_backend") == "local-transformers" and options.get("model"):
        checkpoint = Path(options["model"]).expanduser()
        options["model"] = str((checkpoint if checkpoint.is_absolute() else path.parent / checkpoint).resolve())
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

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def problem_paths(level: int, root: Path = ROOT):
    directory = root / "KernelBench" / f"level{level}"
    paths = {int(p.name.split("_")[0]): p for p in directory.glob("[0-9]*_*.py")}
    if not paths:
        raise ValueError(f"No problems found in {directory}")
    return dict(sorted(paths.items()))


def select_problems(level: int, start: int = 1, end: int | None = None, problem_ids=None, root: Path = ROOT, *, suite="full", suite_role="scored", track="all"):
    paths = problem_paths(level, root)
    end = max(paths) if end is None else end
    selected = list(problem_ids) if problem_ids is not None else list(range(start, end + 1))
    missing = sorted(set(selected) - paths.keys())
    if not selected or missing:
        raise ValueError(f"Invalid level {level} problem selection; missing IDs: {missing}")
    if suite != "full":
        from .suites import SUITES, suite_rows

        if suite not in SUITES:
            raise ValueError(f"Unknown suite: {suite}")
        rows = suite_rows(level, suite_role, suite=suite, track=track)
        selected = [pid for pid in selected if pid in rows]
        for pid in selected:
            if (paths[pid].name != rows[pid]["reference"]
                    or hashlib.sha256(paths[pid].read_bytes()).hexdigest() != rows[pid]["reference_sha256"]):
                raise ValueError(f"Suite reference snapshot mismatch for level {level} problem {pid}")
    elif suite_role != "scored":
        raise ValueError("--suite-role requires --suite kernelbench105")
    elif track != "all":
        raise ValueError("Track selection requires a parameterized suite")
    if not selected:
        raise ValueError("No problems remain in the requested suite, role and range")
    return {pid: paths[pid] for pid in sorted(set(selected))}


def kernel_name(level: int, pid: int, sid: int = 0):
    return f"level_{level}_problem_{pid}_sample_{sid}_kernel.py"


def safe_name(value: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Run name must start with a letter/digit and contain only letters, digits, _, . or -")
    return value


def fingerprint(payload: dict):
    # Request IDs and local build paths are transport details, not evaluation inputs.
    content = {k: v for k, v in payload.items() if k not in {"request_id", "build_root"}}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def evaluator_digest():
    digest = hashlib.sha256()
    for name in (
        "src/eval.py", "src/utils.py", "src/validation.py", "src/reliability.py", "requirements.txt", "requirements-neuron-worker.txt",
        "kb_workflows/evaluate_task.py", "kb_workflows/measurement.py", "kb_workflows/baseline_variants.py",
        "kb_workflows/common.py", "kb_workflows/config.py", "kb_workflows/engine.py",
        "kb_workflows/transports.py", "kb_workflows/paper_analysis.py",
        "kb_workflows/harbor.py", "kb_workflows/harbor_verifier.py",
        "kb_workflows/harbor_budget.py",
        "kb_workflows/resource_budgets.py", "kb_workflows/local_model.py",
        "kb_workflows/devices.py", "kb_workflows/neuron_device.py", "kb_workflows/neuron_worker.py", "kb_workflows/neuron_profile.py",
        "kb_workflows/gpu.py", "kb_workflows/gpu_profiles.toml",
        "kb_workflows/suites.py", "kb_workflows/suites/kernelbench105.json",
        "kb_workflows/evidence.py", "kb_workflows/numerical.py", "kb_workflows/strict_validation.py",
        "kb_workflows/determinism.py", "kb_workflows/tool_checks.py", "kb_workflows/tool_runner.py",
        "kb_workflows/verification_contracts.json",
        "kb_workflows/aggregation.py",
        "kb_workflows/contract_specs.py",
        "kb_workflows/paired_statistics.py", "kb_workflows/paired_task.py", "kb_workflows/final_verification.py",
        "kb_workflows/calibration.py", "kb_workflows/headroom.py", "kb_workflows/performance.py",
        "kb_workflows/production.py", "kb_workflows/production_ops.py", "kb_workflows/production_specs.py",
        "kb_workflows/suites/production41.json",
        "kb_workflows/baseline_rewrites.py", "kb_workflows/baseline_rewrites.json",
        "kb_workflows/agent_loop.py", "kb_workflows/agent_models.py", "kb_workflows/experiment.py",
        "kb_workflows/profile_task.py", "kb_workflows/agent_task.py", "kb_workflows/generation.py",
    ):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def failure(message: str, **metadata):
    return {
        "compiled": False,
        "correctness": False,
        "runtime": -1.0,
        "runtime_stats": {},
        "peak_memory": -1.0,
        "memory_stats": {},
        "metadata": {"error": message, **metadata},
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool, type(None))):
        return value
    return str(value)


def apply_score_gate(results, run_dir):
    """Bind legacy performance metrics to current measurement and source evidence."""
    run_dir = Path(run_dir)
    scored = read_json(run_dir / "eval_results_scored.json", {})
    config = read_json(run_dir / "run_config.json", {})
    standard = read_json(run_dir / "eval_results.json", {})
    hidden = read_json(run_dir / "eval_results_hidden.json", {})
    digest = None
    gated = {}
    auxiliary = {name: read_json(run_dir / name, {}) for name in (
        "final_manifest.json", "final_results.json", "paired_results.json", "generation_budget.json",
        "generation_attempts.json", "generation_summary.json")}
    auxiliary_hashes = {name: fingerprint(value) for name, value in auxiliary.items()}
    for key, raw in results.items():
        options = raw.get("metadata", {}).get("evaluation_config", {})
        capped = any(config.get(name) for name in ("agent_token_limit", "agent_output_token_limit", "agent_device_seconds"))
        required = options.get("score_gate_required", config.get("score_gate_required", False)) or options.get("anti_cheat", False) or capped
        if not required:
            gated[key] = raw
            continue
        if digest is None:
            digest = evaluator_digest()
        entry = scored.get(key, {})
        metadata = entry.get("metadata", {})
        labels = [metadata.get(field, "") for field in
                  ("score_kernel_name", "score_hardware", "score_baseline_name", "score_reference_name")]
        valid_labels = all(isinstance(label, str) and label not in {"", ".", ".."}
                           and Path(label).name == label for label in labels)
        valid = False
        if valid_labels:
            kernel_name_, hardware, baseline_name, reference_name = labels
            kernel = run_dir / kernel_name_
            baselines = read_json(ROOT / "results/timing" / hardware / f"{baseline_name}.json", {})
            baseline = baselines.get(f"level{metadata.get('score_level')}", {}).get(reference_name, {})
            secret = hidden.get(key, {}) if metadata.get("score_hidden_required") else standard.get(key, {})
            level, pid = metadata.get("score_level"), metadata.get("score_problem_id")
            reference = ROOT / "KernelBench" / f"level{level}" / reference_name
            hidden_path = ROOT / "hidden_tests" / f"level{level}" / f"{pid}_hidden.py"
            hidden_source_matches = (not metadata.get("score_hidden_required") or (
                hidden_path.is_file() and secret.get("metadata", {}).get("evaluation_config", {}).get("hidden_source_sha256")
                == hashlib.sha256(hidden_path.read_bytes()).hexdigest()))
            valid = (metadata.get("score_policy") == "measurement_and_compatibility_v1"
                     and metadata.get("score_gate_evaluator_sha256") == digest and kernel.is_file()
                     and metadata.get("kernel_sha256") == fingerprint({"source": kernel.read_text()})
                     and metadata.get("score_standard_sha256") == fingerprint(standard.get(key, {}))
                     and metadata.get("score_hidden_sha256") == fingerprint(secret)
                     and metadata.get("score_baseline_sha256") == fingerprint(baseline)
                     and reference.is_file() and metadata.get("reference_sha256") == fingerprint({"source": reference.read_text()})
                     and hidden_source_matches and metadata.get("score_run_config_sha256") == fingerprint(config))
            final_required = options.get("final_repeats", config.get("final_repeats", 0))
            paired_required = options.get("paired_timing", config.get("paired_timing", False))
            required_auxiliary = (["final_manifest.json", "final_results.json"] if final_required else []) + (["paired_results.json"] if paired_required else [])
            if capped:
                required_auxiliary += ["generation_budget.json", "generation_attempts.json", "generation_summary.json"]
            valid = valid and all(metadata.get("score_auxiliary_files", {}).get(name) == auxiliary_hashes[name] for name in required_auxiliary)
            for option in ("problem_specs", "device_calibration", "task_dimensions", "determinism_policy_file"):
                declared = config.get(option)
                if declared:
                    path = Path(declared)
                    valid = (valid and path.is_file() and metadata.get("score_external_files", {}).get(option)
                             == fingerprint(read_json(path)))
            if valid and final_required:
                final_entry = auxiliary["final_manifest.json"].get("slots", {}).get(key, {})
                source_hash = hashlib.sha256(kernel.read_bytes()).hexdigest()
                frozen = run_dir / "final_artifacts" / (source_hash + ".py")
                valid = (final_entry.get("source_sha256") == source_hash and frozen.is_file()
                         and hashlib.sha256(frozen.read_bytes()).hexdigest() == source_hash)
        gated[key] = {**raw, "correctness": bool(valid and entry.get("correctness") and raw.get("correctness")),
                      "metadata": {**raw.get("metadata", {}), "score_gate_applied": True,
                                   "score_gate_status": metadata.get("score_gate_status") if valid else "missing_or_stale_score_evidence"}}
    return gated

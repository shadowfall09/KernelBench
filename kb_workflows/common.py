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
        from .benchmarks.suites import SUITES, suite_rows

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
    """Hash trusted source/data by stable relative path, without a growing allowlist."""
    digest = hashlib.sha256()
    paths = [ROOT / "requirements.txt", ROOT / "requirements-neuron-worker.txt"]
    for folder in ("src", "kb_workflows", "scripts"):
        paths.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json", ".toml"}
                     and not {"__pycache__", "unit_tests"} & set(path.parts))
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
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
        capped = any(config.get(name) for name in ("agent_token_limit", "agent_output_token_limit", "agent_device_seconds", "agent_gpu_busy_seconds"))
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
            if valid and config.get("evidence_level") == "L2":
                from .config import RunConfig
                from .provenance.counter_task import score_checks

                try:
                    pid_, sid_ = map(int, key.split("_"))
                    checks = score_checks(RunConfig(**config), pid_, sid_)
                    saved = metadata.get("score_measurement_checks", {})
                    valid = all(item["status"] == "passed" and saved.get(name) == item for name, item in checks.items())
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    valid = False
            if valid and config.get("agent_gpu_busy_seconds"):
                from .agents.resource_budgets import gpu_artifact_hashes

                try:
                    pid_, sid_ = map(int, key.split("_"))
                    snapshot = auxiliary["generation_budget.json"]["slots"][key]["resource_budgets"]["gpu_busy_quota"]
                    hashes = gpu_artifact_hashes(run_dir, pid_, sid_, snapshot)
                    valid = all(metadata.get("score_auxiliary_files", {}).get(name) == value for name, value in hashes.items())
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    valid = False
            if valid and config.get("generation") == "harbor" and config.get("agent_output_token_limit"):
                from .runtime.harbor_budget import quota_artifact_hashes

                try:
                    pid_, sid_ = map(int, key.split("_"))
                    summary = auxiliary["generation_summary.json"][key]
                    hashes = quota_artifact_hashes(run_dir, pid_, sid_, summary, level=config["level"])
                    valid = all(metadata.get("score_auxiliary_files", {}).get(name) == value for name, value in hashes.items())
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    valid = False
            for option in ("problem_specs", "device_calibration", "input_calibration", "precision_calibration", "reference_population", "task_dimensions", "determinism_policy_file", "regrade_manifest"):
                declared = config.get(option)
                if declared:
                    path = Path(declared)
                    valid = (valid and path.is_file() and metadata.get("score_external_files", {}).get(option)
                             == fingerprint(read_json(path)))
            if config.get("regrade_manifest"):
                valid = valid and metadata.get("score_regrade_measurements_sha256") == fingerprint(read_json(run_dir / "regrade_measurements.json", {}))
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

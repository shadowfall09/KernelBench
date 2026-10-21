from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_ARCH = {
    "A10G": "Ampere",
    "A100": "Ampere",
    "A100-80GB": "Ampere",
    "RTX_A6000": "Ampere",
    "L40S": "Ada",
    "L4": "Ada",
    "H100": "Hopper",
    "H200": "Hopper",
    "T4": "Turing",
}


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


def select_problems(level: int, start: int = 1, end: int | None = None, problem_ids=None, root: Path = ROOT):
    paths = problem_paths(level, root)
    end = max(paths) if end is None else end
    selected = list(problem_ids) if problem_ids is not None else list(range(start, end + 1))
    missing = sorted(set(selected) - paths.keys())
    if not selected or missing:
        raise ValueError(f"Invalid level {level} problem selection; missing IDs: {missing}")
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
        "src/eval.py", "src/utils.py", "src/validation.py", "src/reliability.py", "requirements.txt",
        "kb_workflows/evaluate_task.py", "kb_workflows/measurement.py", "kb_workflows/baseline_variants.py",
        "kb_workflows/common.py", "kb_workflows/config.py", "kb_workflows/engine.py",
        "kb_workflows/transports.py", "kb_workflows/paper_analysis.py",
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
    for key, raw in results.items():
        options = raw.get("metadata", {}).get("evaluation_config", {})
        required = options.get("score_gate_required", config.get("score_gate_required", False)) or options.get("anti_cheat", False)
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
        gated[key] = {**raw, "correctness": bool(valid and entry.get("correctness") and raw.get("correctness")),
                      "metadata": {**raw.get("metadata", {}), "score_gate_applied": True,
                                   "score_gate_status": metadata.get("score_gate_status") if valid else "missing_or_stale_score_evidence"}}
    return gated

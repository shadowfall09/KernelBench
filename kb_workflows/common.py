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
    for name in ("src/eval.py", "src/utils.py", "kb_workflows/evaluate_task.py", "requirements.txt"):
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

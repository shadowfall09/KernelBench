"""Explicit task repairs, preserving historical source identity and result lineage."""

from __future__ import annotations

import hashlib
import json
import re
from functools import wraps
from pathlib import Path

PROTOCOL = "kernelbench_pro_reference_revisions_v1"
REGISTRY = Path(__file__).with_name("reference_revisions.json")


def registry():
    data = json.loads(REGISTRY.read_text())
    if data.get("protocol") != PROTOCOL or not isinstance(data.get("revisions"), list):
        raise ValueError("Invalid reference revision registry")
    identifiers, latest = set(), {}
    for row in data["revisions"]:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or row["id"] in identifiers or type(row.get("level")) is not int or row["level"] not in {1, 2, 3, 5}
                or type(row.get("problem_id")) is not int or row["problem_id"] <= 0
                or type(row.get("revision")) is not int or row["revision"] < 2
                or not row.get("reason") or not isinstance(row.get("evidence"), list) or not row["evidence"]):
            raise ValueError("Reference revisions require unique IDs, task identities, reasons and evidence")
        for name in ("reference_before_sha256", "reference_sha256", "hidden_before_sha256", "hidden_sha256"):
            if not isinstance(row.get(name), str) or not re.fullmatch("[0-9a-f]{64}", row[name]):
                raise ValueError("Reference revisions require complete before/after source hashes")
        dependencies = row.get("hidden_dependencies", {})
        if (not isinstance(dependencies, dict) or any(not isinstance(name, str)
                or not re.fullmatch(r"kb_workflows/[A-Za-z0-9_]+\.py", name)
                or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                for name, digest in dependencies.items())):
            raise ValueError("Reference revisions require bounded hidden dependency paths and hashes")
        key = row["level"], row["problem_id"]
        previous = latest.get(key)
        if previous and (row["revision"] != previous["revision"] + 1
                         or row["reference_before_sha256"] != previous["reference_sha256"]
                         or row["hidden_before_sha256"] != previous["hidden_sha256"]):
            raise ValueError("Reference revision history is not a continuous source chain")
        if previous is None and row["revision"] != 2:
            raise ValueError("Reference repair history must begin at revision 2")
        identifiers.add(row["id"])
        latest[key] = row
    return data


def binding(level, pid, source):
    history = [row for row in registry()["revisions"] if (row["level"], row["problem_id"]) == (level, pid)]
    if not history:
        return None
    current = history[-1]
    if hashlib.sha256(source.encode()).hexdigest() != current["reference_sha256"]:
        raise ValueError("Task reference does not match its current declared revision")
    for name, expected in current.get("hidden_dependencies", {}).items():
        path = REGISTRY.parent.parent / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Hidden input dependency differs from its declared task revision")
    return {"protocol": PROTOCOL, "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
            "id": current["id"], "revision": current["revision"],
            "reference_sha256": current["reference_sha256"], "hidden_sha256": current["hidden_sha256"],
            "hidden_dependencies": current.get("hidden_dependencies", {}),
            "history": [row["id"] for row in history]}


def validate_payload(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("eval_config"), dict):
        raise ValueError("Task revision validation requires a request and settings dictionary")
    for name in ("level", "problem_id"):
        if name in payload and (type(payload[name]) is not int or payload[name] <= 0):
            raise ValueError("Task revision identities must use positive integers")
    if "ref_arch_src" in payload and not isinstance(payload["ref_arch_src"], str):
        raise ValueError("Task revision source must be a string")
    expected = binding(payload.get("level"), payload.get("problem_id"), payload.get("ref_arch_src", ""))
    settings = payload["eval_config"]
    if settings.get("reference_revision") != expected:
        raise ValueError("Missing, changed or retired task reference revision")
    if expected and settings.get("use_hidden_tests") and settings.get("hidden_source_sha256") != expected["hidden_sha256"]:
        raise ValueError("Hidden tests do not match the declared task reference revision")
    return expected


def record_result(function):
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        expected = validate_payload(payload)
        result = function(payload, *args, **kwargs)
        if expected:
            result.setdefault("metadata", {})["reference_revision"] = expected
        return result
    return wrapped


def issue(result, payload):
    """A current positive result must identify the repaired task it executed."""
    try:
        expected = validate_payload(payload)
    except (KeyError, TypeError, ValueError, OSError):
        return "task_reference_revision_mismatch"
    if expected and result.get("correctness") is True and result.get("metadata", {}).get("reference_revision") != expected:
        return "task_reference_revision_evidence_missing_or_changed"
    return None

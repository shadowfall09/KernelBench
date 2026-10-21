"""Versioned task selections; research labels are not bound certificates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("suites") / "kernelbench105.json"
SUITES = ("full", "kernelbench105")
ROLES = ("scored", "probe", "extended")


def load_manifest():
    data = json.loads(MANIFEST.read_text())
    rows = data["problems"]
    identities = [(row["level"], row["problem_id"]) for row in rows]
    if (len(rows) != 250 or len(set(identities)) != 250
            or any(row["role"] not in ROLES or row["layer"] not in {"I", "F", "D", "unclassified"} for row in rows)
            or sum(row["role"] == "scored" for row in rows) != 105
            or sum(row["role"] == "probe" for row in rows) != 5):
        raise ValueError("Invalid KernelBench-105 manifest")
    return data


def suite_rows(level, role):
    if role not in ROLES:
        raise ValueError(f"Unknown suite role: {role}")
    return {row["problem_id"]: row for row in load_manifest()["problems"]
            if row["level"] == level and row["role"] == role}


def selection_receipt(config):
    if config.suite == "full":
        return {"name": "full", "role": "scored", "layer_certificates": False}
    return {"name": config.suite, "role": config.suite_role,
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "seed": load_manifest()["selection_seed"], "layer_certificates": False,
            "input_calibration": "not_implemented", "historical_ranking_reproduced": False,
            "purpose": "research_selection; probe and extended runs reported separately"}

"""Versioned task selections; research labels are not bound certificates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("suites") / "kernelbench105.json"
PRODUCTION_MANIFEST = Path(__file__).with_name("suites") / "production41.json"
SUITES = ("full", "kernelbench105", "production41")
ROLES = ("scored", "probe", "extended")


def load_manifest(suite="kernelbench105"):
    if suite == "production41":
        data = json.loads(PRODUCTION_MANIFEST.read_text())
        rows = data["problems"]
        if (len(rows) != 41 or {row["problem_id"] for row in rows} != set(range(1, 42))
                or any(row["level"] != 5 or row["role"] != "scored" for row in rows)):
            raise ValueError("Invalid production41 manifest")
        return data
    if suite != "kernelbench105":
        raise ValueError("Unknown selection manifest")
    data = json.loads(MANIFEST.read_text())
    from .reference_revisions import REGISTRY, registry

    revisions = registry()["revisions"]
    if (data.get("version") != 2 or data.get("reference_revision_registry_sha256") != hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
            or data.get("reference_snapshot") != "base_snapshot_plus_declared_reference_revisions"):
        raise ValueError("Selection manifest has no matching current reference revision registry")
    rows = data["problems"]
    identities = [(row["level"], row["problem_id"]) for row in rows]
    if (len(rows) != 250 or len(set(identities)) != 250
            or any(row["role"] not in ROLES or row["layer"] not in {"I", "F", "D", "unclassified"} for row in rows)
            or sum(row["role"] == "scored" for row in rows) != 105
            or sum(row["role"] == "probe" for row in rows) != 5):
        raise ValueError("Invalid KernelBench-105 manifest")
    for revision in revisions:
        matching = [row for row in rows if (row["level"], row["problem_id"]) == (revision["level"], revision["problem_id"])]
        # Only the latest repair in each task's chain is active.
        if any(later["level"] == revision["level"] and later["problem_id"] == revision["problem_id"]
               and later["revision"] > revision["revision"] for later in revisions):
            continue
        if matching and (matching[0].get("reference_revision") != revision["id"] or matching[0]["reference_sha256"] != revision["reference_sha256"]):
            raise ValueError("Selection reference repair differs from the declared task revision")
    return data


def suite_rows(level, role, suite="kernelbench105", track="all"):
    if role not in ROLES:
        raise ValueError(f"Unknown suite role: {role}")
    return {row["problem_id"]: row for row in load_manifest(suite)["problems"]
            if row["level"] == level and row["role"] == role and (track == "all" or row.get("track") == track)}


def selection_receipt(config):
    result = _selection_receipt(config)
    from .reference_population import receipt

    if population := receipt(config):
        result["reference_population"] = population
    return result


def _selection_receipt(config):
    if config.suite == "full":
        from .reference_revisions import REGISTRY, registry

        return {"name": "full", "role": "scored", "layer_certificates": False,
                "reference_revision_registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
                "reference_revisions": [row["id"] for row in registry()["revisions"] if row["level"] == config.level]}
    if config.suite == "production41":
        return {"name": "production41", "role": config.suite_role, "layer_certificates": False,
                "manifest_sha256": hashlib.sha256(PRODUCTION_MANIFEST.read_bytes()).hexdigest(),
                "input_calibration": "role_domains_and_memory_plan_v1", "historical_ranking_reproduced": False,
                "dtype_instances_scored_separately": True, "reachability_results_collected": False,
                "profile": config.task_profile, "memory_budget_mb": config.task_memory_mb,
                "track": config.task_track,
                "purpose": "production_parameterized_families; design_labels_provisional"}
    manifest = load_manifest()
    return {"name": config.suite, "role": config.suite_role, "version": manifest["version"],
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "seed": manifest["selection_seed"], "layer_certificates": False,
            "selection_reference_snapshot": manifest["selection_reference_snapshot"],
            "reference_revision_registry_sha256": manifest["reference_revision_registry_sha256"],
            "input_calibration": "not_implemented", "historical_ranking_reproduced": False,
            "purpose": "research_selection; probe and extended runs reported separately"}

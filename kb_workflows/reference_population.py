"""Preregister reference eligibility before any candidate or performance result."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .common import atomic_json, evaluator_digest, fingerprint, read_json, select_problems
from .input_calibration import read_bundle, validate_record
from .input_profiles import recipes

PROTOCOL = "kernelbench_pro_reference_population_v1"
SNAPSHOT = "reference_population.json"


def declared(config):
    return select_problems(config.level, config.start, config.end, suite=config.suite,
                           suite_role=config.suite_role, track=config.task_track)


def _selection(config, paths):
    return {"level": config.level, "precision": config.precision, "suite": config.suite,
            "suite_role": config.suite_role, "task_track": config.task_track,
            "references": {str(pid): hashlib.sha256(path.read_bytes()).hexdigest() for pid, path in paths.items()}}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _reasons(record):
    """Only complete numerical/signal failures may exclude a reference."""
    from .input_calibration import _qualified
    from .numerical_audit import _signal_valid

    if not record["qualified"] and len(record["attempts"]) != len(recipes(record["problem_id"], record["level"])):
        raise ValueError("Unqualified reference has untested distributions; population is unresolved")
    summaries = []
    for attempt in record["attempts"]:
        reasons = set()
        rows = attempt["cases"]
        for row in rows:
            case_reasons = set()
            gate, signal = row.get("reference_gate"), row.get("signal")
            # Resource/worker failures are unresolved, never semantic exclusions.
            if (row.get("errors") != [] or row.get("status") not in {"passed", "failed"}
                    or not isinstance(gate, dict) or type(gate.get("passed")) is not bool
                    or type(gate.get("reference_qualified")) is not bool
                    or not isinstance(gate.get("issues"), list) or not isinstance(gate.get("reference_issues"), list)
                    or gate.get("all_elements_required") is not True or gate.get("oracle_dtype") != "float64/complex128"
                    or not _signal_valid(signal, 0.0)
                    or row["status"] != ("passed" if gate["passed"] else "failed")
                    or any(not _sha(row.get(key)) for key in ("input_probe_sha256", "output_probe_sha256"))
                    or type(row.get("estimated_host_bytes")) is not int
                    or not 0 < row["estimated_host_bytes"] <= record["oracle_memory_mb"] * 1024**2):
                raise ValueError("Incomplete reference numerical evidence; population is unresolved")
            if not gate["passed"] or not gate["reference_qualified"]:
                case_reasons.add("reference_precision_failed")
            if not signal["informative"]:
                case_reasons.add("reference_output_noninformative")
            if any(leaf.get("kind") != "discrete_exact" and leaf["fraction_above_atol"] < record["selection_policy"]["minimum_fraction"]
                   for leaf in signal["leaves"]):
                case_reasons.add("insufficient_nonzero_signal")
            if not case_reasons and not _qualified(row, record["precision_contract"], record["selection_policy"]["minimum_fraction"]):
                raise ValueError("Inconsistent reference precision observations; population is unresolved")
            spec = record["oracle_spec"]
            if spec["family"] in {"composite", "model_block"}:
                from .composite_specs import receipt_valid as composite_receipt
                from .model_specs import receipt_valid as model_receipt

                check = model_receipt if spec["family"] == "model_block" else composite_receipt
                if not isinstance(row.get("source_device"), str) or not check(row.get("oracle_receipt"), spec,
                        precision=record["precision"], source_device=row["source_device"]):
                    raise ValueError("Incomplete reference program/mask receipt; population is unresolved")
            reasons.update(case_reasons)
        if len({row["input_probe_sha256"] for row in rows}) != len(rows):
            reasons.add("input_samples_not_distinct")
        if len({row["output_probe_sha256"] for row in rows}) < 2:
            reasons.add("output_samples_not_distinct")
        summaries.append({"recipe": attempt["recipe"], "reasons": sorted(reasons),
                          "case_sha256": [fingerprint(row) for row in rows]})
    if record["qualified"] != any(not row["reasons"] for row in summaries):
        raise ValueError("Reference eligibility differs from its complete observations")
    return summaries


def build(config, *, paths=None):
    """Read existing reference records only; no Torch, execution, timing or model."""
    if not config.input_calibration or config.level not in {1, 2, 3}:
        raise ValueError("Reference populations require original L1/L2/L3 input calibration")
    bundle = read_bundle(config.input_calibration)
    if bundle.get("errors") or bundle.get("population_manifest"):
        raise ValueError("Unresolved calibration workers or a delegated bundle cannot define a population")
    paths, ledger = declared(config) if paths is None else paths, []
    for pid, path in paths.items():
        key = f"{config.level}/{pid}/{config.precision}"
        record = bundle["records"].get(key)
        if not isinstance(record, dict) or (record.get("level"), record.get("problem_id"), record.get("precision")) != (config.level, pid, config.precision):
            raise ValueError(f"Reference population lacks a declared calibration record: {key}")
        validate_record(record, path.read_text(), require_qualified=False)
        ledger.append({"key": key, "problem_id": pid, "qualified": record["qualified"],
                       "record_sha256": record["receipt_sha256"], "selected_recipe": record["selected_recipe"],
                       "attempts": _reasons(record)})
    qualified = [row["problem_id"] for row in ledger if row["qualified"]]
    if not qualified:
        raise ValueError("No qualified references; an empty population cannot produce a score")
    result = {"protocol": PROTOCOL, "selection": _selection(config, paths), "evaluator_sha256": evaluator_digest(),
              "calibration_sha256": fingerprint(bundle), "ledger": ledger,
              "qualified": qualified, "excluded": [row["problem_id"] for row in ledger if not row["qualified"]],
              "selection_basis": "reference_only_before_candidates; first_qualified_distribution",
              "post_candidate_failures": "retain_all_declared_samples_with_zero_score",
              "candidate_executed": False, "timings_collected": False}
    result["manifest_sha256"] = fingerprint(result)
    return result


def _validate_shape(manifest):
    if (not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL
            or manifest.get("manifest_sha256") != fingerprint({k: v for k, v in manifest.items() if k != "manifest_sha256"})
            or manifest.get("evaluator_sha256") != evaluator_digest()
            or manifest.get("candidate_executed") is not False or manifest.get("timings_collected") is not False
            or not _sha(manifest.get("calibration_sha256"))
            or manifest.get("selection_basis") != "reference_only_before_candidates; first_qualified_distribution"
            or manifest.get("post_candidate_failures") != "retain_all_declared_samples_with_zero_score"):
        raise ValueError("Missing, stale or changed reference population")
    selection, ledger = manifest.get("selection", {}), manifest.get("ledger")
    if (not isinstance(selection, dict) or selection.get("level") not in {1, 2, 3}
            or selection.get("precision") not in {"fp32", "fp16", "bf16"}
            or not isinstance(selection.get("references"), dict) or not selection["references"]
            or any(not isinstance(key, str) or not key.isdecimal() or not _sha(value) for key, value in selection["references"].items())
            or not isinstance(ledger, list) or any(not isinstance(row, dict) or type(row.get("problem_id")) is not int
                or type(row.get("qualified")) is not bool or not _sha(row.get("record_sha256")) for row in ledger)
            or [row["problem_id"] for row in ledger] != sorted(map(int, selection["references"]))
            or any(row.get("key") != f"{selection['level']}/{row['problem_id']}/{selection['precision']}" for row in ledger)
            or not manifest.get("qualified")
            or manifest["qualified"] != [row["problem_id"] for row in ledger if row["qualified"]]
            or manifest.get("excluded") != [row["problem_id"] for row in ledger if not row["qualified"]]):
        raise ValueError("Reference population ledger changed its declared denominator")
    return manifest


def manifest_for(config):
    path = getattr(config, "reference_population", "")
    if path:
        manifest = _validate_shape(read_json(Path(path)))
        paths = None
        if config.reference_population_scope == "task":
            selected = declared(config)
            if len(selected) != 1 or next(iter(selected)) not in manifest["qualified"]:
                raise ValueError("Reference population task delegation requires one qualified reference")
            selection = manifest["selection"]
            if any(getattr(config, key) != selection[key] for key in ("level", "precision", "suite", "suite_role", "task_track")):
                raise ValueError("Delegated task changed the original reference population contract")
            paths = select_problems(config.level, problem_ids=map(int, selection["references"]),
                                    suite=config.suite, suite_role=config.suite_role, track=config.task_track)
        if manifest != build(config, paths=paths):
            raise ValueError("Reference population differs from current sources, calibration or declared selection")
        return manifest
    if config.input_calibration:
        bundle = read_bundle(config.input_calibration)
        if "population_manifest" in bundle:
            manifest = _validate_shape(bundle["population_manifest"])
            paths = declared(config)
            if len(paths) != 1:
                raise ValueError("A delegated population bundle is restricted to one qualified task")
            pid, source = next(iter(paths.items()))
            key = f"{config.level}/{pid}/{config.precision}"
            if (bundle.get("delegated_task") != key or pid not in manifest["qualified"]
                    or manifest["selection"].get("level") != config.level or manifest["selection"].get("precision") != config.precision
                    or manifest["selection"]["references"].get(str(pid)) != hashlib.sha256(source.read_bytes()).hexdigest()):
                raise ValueError("Delegated reference population belongs to another task/source")
            record = bundle["records"].get(key)
            validate_record(record, source.read_text())
            if record["receipt_sha256"] != next(row["record_sha256"] for row in manifest["ledger"] if row["problem_id"] == pid):
                raise ValueError("Delegated calibration differs from frozen reference population")
            return manifest
    return None


def select_run_problems(config):
    paths, manifest = declared(config), manifest_for(config)
    snapshot = config.run_dir / SNAPSHOT
    if snapshot.exists() or snapshot.is_symlink():
        if snapshot.is_symlink() or manifest is None or read_json(snapshot) != _run_snapshot(config, paths, manifest):
            raise ValueError("Frozen run reference population changed; use a new run")
    if manifest is None:
        return paths
    return {pid: path for pid, path in paths.items() if pid in manifest["qualified"]}


def _run_snapshot(config, paths, manifest):
    return {"manifest": manifest, "active_problem_ids": [pid for pid in paths if pid in manifest["qualified"]],
            "num_samples": config.num_samples, "scope": config.reference_population_scope,
            "delegated_bundle": not bool(config.reference_population)}


def freeze_run(config):
    """Persist before generation/export/evaluation, never adopt an existing run."""
    manifest = manifest_for(config)
    select_run_problems(config)
    if manifest is None or config.dry_run:
        return manifest
    snapshot = config.run_dir / SNAPSHOT
    if not snapshot.exists():
        if (any(config.run_dir.glob("level_*_kernel.py")) or any((config.run_dir / name).exists() for name in (
                "work", "generation_summary.json", "generation_attempts.json", "eval_results.json", "eval_results_hidden.json",
                "final_manifest.json", "paired_results.json", "paper_analysis.json", "harbor_export.json"))):
            raise ValueError("Cannot choose a reference population after candidate artifacts or results exist")
        atomic_json(snapshot, _run_snapshot(config, declared(config), manifest))
    return manifest


def binding(config, pid):
    manifest = manifest_for(config)
    if manifest is None:
        return None
    if pid not in select_run_problems(config):
        raise ValueError("Task is outside the preregistered qualified reference population")
    return {"manifest": manifest, "active_key": f"{config.level}/{pid}/{config.precision}"}


def validate_inline(payload, record):
    binding_ = payload["eval_config"].get("reference_population")
    if binding_ is None:
        return None
    if not isinstance(binding_, dict):
        raise ValueError("Invalid inline reference population")
    manifest = _validate_shape(binding_.get("manifest"))
    level, pid, precision = payload["level"], payload["problem_id"], payload["eval_config"]["precision"]
    key = f"{level}/{pid}/{precision}"
    if (binding_.get("active_key") != key or pid not in manifest["qualified"]
            or (manifest["selection"]["level"], manifest["selection"]["precision"]) != (level, precision)
            or manifest["selection"]["references"].get(str(pid)) != hashlib.sha256(payload["ref_arch_src"].encode()).hexdigest()
            or not isinstance(record, dict) or record.get("receipt_sha256") != next(row["record_sha256"] for row in manifest["ledger"] if row["problem_id"] == pid)):
        raise ValueError("Task calibration is outside the frozen reference population")
    return manifest["manifest_sha256"]


def receipt(config):
    manifest = manifest_for(config)
    if manifest is None:
        return None
    return {**{key: manifest[key] for key in ("protocol", "manifest_sha256", "calibration_sha256", "qualified", "excluded", "ledger")},
            "active_problem_ids": [pid for pid in declared(config) if pid in manifest["qualified"]]}

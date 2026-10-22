"""Content-bound policy history and supervisor receipts, without model execution."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .common import atomic_json, evaluator_digest, fingerprint, read_json
from .local_model import checkpoint_digest
from .resource_budgets import validate_snapshot

PROTOCOL = "kernelbench_pro_training_lineage_v1"
SUPERVISOR_PROTOCOL = "kernelbench_pro_training_supervisor_v1"


def seal(document):
    return {**document, "receipt_sha256": fingerprint({"document": document})}


def checked(document, protocol):
    if (not isinstance(document, dict) or document.get("protocol") != protocol or document.get("receipt_sha256") !=
            fingerprint({"document": {key: value for key, value in document.items() if key != "receipt_sha256"}})):
        raise ValueError("Missing or modified training provenance receipt")
    return document


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_file(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Training evidence must be a regular file inside its declared run")
    return str(path.relative_to(root))


def emit(plan, journal, result, *, attempt_id, execution_kind, supervisor_nonce=None):
    """Archive the exact reward/journal prefix used by the saved final policy."""
    from .rl_rewards import validate_reward

    root = Path(plan["output_directory"]).resolve()
    plan_file = root / "training_plan.json"
    if plan_file.exists() and read_json(plan_file) != plan:
        raise ValueError("Saved training plan differs from the current run")
    atomic_json(plan_file, plan)
    directory = root / "lineage"
    directory.mkdir(exist_ok=True)
    journal_file = directory / f"{attempt_id}-journal.json"
    if journal_file.exists():
        raise ValueError("A final policy lineage attempt cannot be overwritten")
    atomic_json(journal_file, journal.document)
    files = {}

    def add(path):
        name = relative_file(root, path)
        files[name] = file_sha(root / name)

    add(plan_file)
    add(journal_file)
    for group in journal.document["groups"]:
        if not group.get("reward_path"):
            continue
        path = Path(group["reward_path"])
        reward = validate_reward(path)
        if reward["receipt_sha256"] != group.get("reward_sha256"):
            raise ValueError("Reward history and saved model disagree")
        add(path)
        for name in reward["source_files"]:
            add(path.parent / name)
        for item in reward.get("scaling_certificates", {}).values():
            if item.get("path"):
                from .scaling_certificate import validate as validate_scaling

                certificate_path = path.parent / item["path"]
                certificate = validate_scaling(certificate_path)
                add(certificate_path)
                for name in certificate["proof_files"]:
                    add(certificate_path.parent / name)
        from .common import kernel_name

        level, pid = map(int, reward["task_key"].split("/"))
        for sid, source in enumerate(reward["source_sha256"]):
            if source:
                add(path.parent / kernel_name(level, pid, sid))
        final = path.parent / "final_artifacts"
        for candidate in final.glob("*.py"):
            add(candidate)
    active = list(journal.active_group_ids)
    indexed = {group["id"]: group for group in journal.document["groups"]}
    if (len(set(active)) != len(active) or any(group not in indexed or
            type(indexed[group].get("optimizer_step_after")) is not int
            or indexed[group].get("status") != "rewarded" or not indexed[group].get("reward_path") for group in active)):
        raise ValueError("Final policy lacks a valid optimizer history")
    step = result.get("global_step")
    if type(step) is not int or step != len(active) or any(indexed[group]["optimizer_step_after"] != i + 1 for i, group in enumerate(active)):
        raise ValueError("Final trainer step does not match its active optimizer path")
    receipt = seal({"protocol": PROTOCOL, "training_root": str(root), "attempt_id": attempt_id,
                    "execution_kind": execution_kind, "supervisor_nonce": supervisor_nonce,
                    "software_versions": result.get("software_versions", {}),
                    "plan_sha256": plan["plan_sha256"], "evaluator_sha256": plan["evaluator_sha256"],
                    "rl_implementation_sha256": plan["rl_implementation_sha256"],
                    "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"],
                    "checkpoint_sha256": result["final_model_sha256"], "global_step": step,
                    "active_group_ids": active, "consumed_groups": len(journal.document["groups"]),
                    "journal_file": str(journal_file.relative_to(root)), "proof_files": files,
                    "worker_status": result["status"], "provenance_scope": "verifier_owned_content_bindings; not_cryptographic_or_hardware_attestation"})
    target = directory / f"{attempt_id}.json"
    atomic_json(target, receipt)
    atomic_json(root / "final_lineage.json", receipt)
    return target, receipt


def supervisor(plan, nonce, ledger, code, worker):
    root = Path(plan["output_directory"])
    receipt = seal({"protocol": SUPERVISOR_PROTOCOL, "plan_sha256": plan["plan_sha256"], "nonce": nonce,
                    "worker_exit_code": code, "worker_result_sha256": fingerprint({"document": worker}),
                    "lineage_sha256": worker.get("lineage_sha256"), "ledger": ledger,
                    "physical_preemption_certified": False})
    atomic_json(root / "supervisor" / f"{nonce}.json", receipt)
    return receipt


def validate(checkpoint, *, receipt_path=None, allow_injected=False, require_supervisor=True, current=True):
    """Historical validation preserves artifacts; current qualification is checked separately."""
    from .rl_plan import implementation_digest, validate_plan

    checkpoint = Path(checkpoint).resolve()
    path = Path(receipt_path) if receipt_path else checkpoint.parent / "final_lineage.json"
    receipt = checked(read_json(path), PROTOCOL)
    root = Path(receipt["training_root"])
    for name, digest in receipt["proof_files"].items():
        if relative_file(root, name) != name or file_sha(root / name) != digest:
            raise ValueError("Archived training evidence changed")
    plan = read_json(root / "training_plan.json")
    if (plan.get("plan_sha256") != receipt["plan_sha256"] or plan.get("plan_sha256") !=
            fingerprint({key: value for key, value in plan.items() if key != "plan_sha256"})
            or plan.get("initial_checkpoint_sha256") != receipt["initial_checkpoint_sha256"]
            or checkpoint_digest(checkpoint) != receipt["checkpoint_sha256"]):
        raise ValueError("Policy checkpoint and training provenance differ")
    if current:
        validate_plan(plan)
        if receipt["evaluator_sha256"] != evaluator_digest() or receipt["rl_implementation_sha256"] != implementation_digest():
            raise ValueError("Policy training used a different evaluator/implementation")
    injected = receipt.get("execution_kind") != "native_trl"
    if injected and not allow_injected:
        raise ValueError("Injected trainer/verifier artifacts cannot establish a native training effect")
    journal = read_json(root / receipt["journal_file"])
    groups = journal["groups"]
    indexed = {group["id"]: group for group in groups}
    active = receipt["active_group_ids"]
    if (len(indexed) != len(groups) or len(active) != receipt["global_step"] or len(set(active)) != len(active)
            or receipt["consumed_groups"] != len(groups) or len(groups) > plan["training"]["max_steps"]
            or any(group not in indexed or indexed[group].get("optimizer_step_after") != i + 1
                   or indexed[group].get("status") != "rewarded" or not indexed[group].get("reward_path") for i, group in enumerate(active))):
        raise ValueError("Archived optimizer path is inconsistent")
    if current:
        from .rl_rewards import validate_reward

        for group in groups:
            if group.get("reward_path"):
                reward = validate_reward(group["reward_path"])
                if reward["receipt_sha256"] != group["reward_sha256"] or reward["plan_sha256"] != plan["plan_sha256"]:
                    raise ValueError("A training reward belongs to a different source or plan")
    supervised = False
    nonce = receipt.get("supervisor_nonce")
    if nonce:
        if not isinstance(nonce, str) or not re.fullmatch("[0-9a-f]{32}", nonce):
            raise ValueError("Invalid supervisor nonce")
        wall = checked(read_json(root / "supervisor" / f"{nonce}.json"), SUPERVISOR_PROTOCOL)
        supervised = (wall.get("nonce") == nonce and wall.get("plan_sha256") == plan["plan_sha256"]
                      and wall.get("lineage_sha256") == receipt["receipt_sha256"]
                      and validate_snapshot(wall.get("ledger"), plan["training"]["wall_seconds"], "training_process_wall_seconds"))
    if require_supervisor and not supervised:
        raise ValueError("Policy lacks a complete compliant parent wall-time receipt")
    return {"receipt": receipt, "plan": plan, "journal": journal, "native_execution": not injected,
            "supervised": supervised, "current_protocol": current, "formal_comparison_eligible": not injected and supervised and current}


def comparison_contract(plan):
    """Reward construction is the treatment; optimizer, data and budgets must agree."""
    training = dict(plan["training"])
    for key in ("checkpoint", "reward_mode", "seed", "certificate_bonus", "certificate_min_score", "scaling_contracts"):
        training.pop(key, None)
    verifier = dict(plan["verifier_config"])
    for key in ("run_name", "runs_dir", "stage", "resume", "model", "small_model", "agent_backend"):
        verifier.pop(key, None)
    for name, digest in plan.get("verifier_inputs", {}).items():
        verifier[name] = {"content_sha256": digest}
    return {"training": training, "verifier": verifier, "tasks": plan["tasks"], "splits": plan["splits"],
            "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"], "trainer": plan["trainer"],
            "evaluator_sha256": plan["evaluator_sha256"], "rl_implementation_sha256": plan["rl_implementation_sha256"]}

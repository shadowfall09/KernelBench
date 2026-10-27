"""Admission reservations and fail-closed accounting for controlled agent actions."""

from __future__ import annotations

import math
import time
import uuid
from functools import wraps

from ..common import failure, fingerprint

PROTOCOL = "agent_resource_reservations_v1"
LEASE_PROTOCOL = "development_device_lease_v1"
CLEANUP_SECONDS = 5.0


class BudgetExhausted(RuntimeError):
    pass


def amount(value, *, integer=False):
    if type(value) is int:
        return value >= 0
    return not integer and type(value) is float and math.isfinite(value) and value >= 0


class Ledger:
    """Unknown in-flight usage burns its full reservation; overruns remain visible."""

    def __init__(self, limit, unit, *, integer=False):
        if not amount(limit, integer=integer):
            raise ValueError("Invalid resource limit")
        self.limit, self.unit, self.integer = limit, unit, integer
        self.entries, self.pending = [], {}

    @property
    def spent(self):
        return sum(entry["charged"] for entry in self.entries)

    @property
    def remaining(self):
        return max(0, self.limit - self.spent - sum(self.pending.values()))

    def reserve(self, maximum):
        if not amount(maximum, integer=self.integer) or maximum <= 0:
            raise ValueError("Reservations must be positive")
        if maximum > self.remaining:
            raise BudgetExhausted(f"{self.unit} budget cannot admit this action")
        nonce = uuid.uuid4().hex
        self.pending[nonce] = maximum
        return nonce

    def settle(self, nonce, observed=None):
        if nonce not in self.pending:
            raise ValueError("Missing or already settled reservation")
        maximum = self.pending.pop(nonce)
        known = amount(observed, integer=self.integer)
        charged = observed if known else maximum
        self.entries.append({"nonce": nonce, "reserved": maximum, "observed": observed if known else None,
                             "charged": charged, "within_reservation": known and observed <= maximum})
        return self.entries[-1]["within_reservation"]

    def snapshot(self):
        return {"protocol": PROTOCOL, "unit": self.unit, "limit": self.limit, "charged": self.spent,
                "remaining": self.remaining, "pending": dict(self.pending), "entries": list(self.entries),
                "compliant": not self.pending and all(row["within_reservation"] for row in self.entries)
                    and self.spent <= self.limit}


def validate_snapshot(snapshot, limit, unit):
    if not isinstance(snapshot, dict) or snapshot.get("protocol") != PROTOCOL or snapshot.get("unit") != unit or snapshot.get("limit") != limit:
        return False
    entries, pending = snapshot.get("entries"), snapshot.get("pending")
    if not isinstance(entries, list) or pending != {}:
        return False
    nonces, spent = set(), 0
    for row in entries:
        if not isinstance(row, dict):
            return False
        nonce = row.get("nonce")
        if (not isinstance(nonce, str) or not nonce or nonce in nonces or not amount(row.get("reserved"))
                or row["reserved"] <= 0
                or not amount(row.get("observed")) or row.get("charged") != row["observed"]
                or row["observed"] > row["reserved"] or row.get("within_reservation") is not True):
            return False
        if unit == "tokens" and any(type(row[key]) is not int for key in ("reserved", "observed", "charged")):
            return False
        nonces.add(nonce)
        spent += row["charged"]
    return (snapshot.get("charged") == spent and snapshot.get("remaining") == max(0, limit - spent)
            and spent <= limit and snapshot.get("compliant") is True)


def device_lease(ledger, timeout, wall_remaining, *, gpu_ledger=None):
    limits = [timeout + CLEANUP_SECONDS, wall_remaining]
    limits += [item.remaining for item in (ledger, gpu_ledger) if item is not None]
    window = min(limits)
    if window <= CLEANUP_SECONDS:
        raise BudgetExhausted("Insufficient development device budget including cleanup allowance")
    nonce = ledger.reserve(window) if ledger is not None else gpu_ledger.reserve(window)
    lease = {"protocol": LEASE_PROTOCOL, "nonce": nonce, "reserved_seconds": window,
             "execution_timeout_seconds": window - CLEANUP_SECONDS,
             "scope": "exclusive_development_worker_wall_lease; not_GPU_busy_or_billing"}
    if gpu_ledger is not None:
        lease["gpu_busy"] = {"nonce": gpu_ledger.reserve(window) if ledger is not None else nonce,
                             "reserved_seconds": window, "limit_seconds": gpu_ledger.limit,
                             "policy": "activity_union_or_confirmed_worker_wall_upper_bound_v1"}
    return lease


def validate_device_lease(payload):
    lease = payload.get("eval_config", {}).get("device_lease")
    if lease is None:
        return None
    timeout = payload["eval_config"].get("timeout")
    if (not isinstance(lease, dict) or lease.get("protocol") != LEASE_PROTOCOL
            or not isinstance(lease.get("nonce"), str) or not lease["nonce"]
            or not amount(lease.get("reserved_seconds")) or not amount(lease.get("execution_timeout_seconds"))
            or not amount(timeout) or not 0 < timeout <= lease["execution_timeout_seconds"]
            or lease["execution_timeout_seconds"] + CLEANUP_SECONDS > lease["reserved_seconds"] + 1e-9):
        raise ValueError("Invalid development resource lease")
    gpu = lease.get("gpu_busy")
    if gpu is not None and (not isinstance(gpu, dict) or not isinstance(gpu.get("nonce"), str) or not gpu["nonce"]
            or gpu.get("reserved_seconds") != lease["reserved_seconds"] or not amount(gpu.get("limit_seconds"))
            or gpu["reserved_seconds"] > gpu["limit_seconds"]
            or gpu.get("policy") != "activity_union_or_confirmed_worker_wall_upper_bound_v1"
            or payload["eval_config"].get("device_backend", "cuda") != "cuda"):
        raise ValueError("Invalid GPU busy reservation")
    return lease


def stamp_device_lease(result, lease, elapsed):
    """Only trusted parents call this, after work/cleanup or confirmed non-execution."""
    receipt = {"protocol": LEASE_PROTOCOL, "nonce": lease["nonce"], "reserved_seconds": lease["reserved_seconds"],
               "elapsed_seconds": elapsed, "method": "trusted_parent_monotonic_wall", "physical_preemption_certified": False}
    if lease.get("gpu_busy") is not None:
        receipt["lease_sha256"] = fingerprint(lease)
    receipt["receipt_sha256"] = fingerprint(receipt)
    result.setdefault("metadata", {})["device_lease_receipt"] = receipt
    if elapsed > lease["reserved_seconds"]:
        result.update(compiled=False, correctness=False)
        result["metadata"]["error"] = "Worker exceeded its reserved device lease, including cleanup"
    return result


def record_device_lease(function):
    """Parent-process timing overwrites any receipt supplied by the candidate child."""
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        lease = validate_device_lease(payload)
        if lease is None:
            return function(payload, *args, **kwargs)
        started = time.monotonic()
        try:
            result = function(payload, *args, **kwargs)
        except Exception as exc:
            # In particular, a failed cleanup can leave usage continuing after
            # this parent returns. A short elapsed time cannot certify it.
            return failure(str(exc), error_name=type(exc).__name__, resource_usage_unknown=True)
        elapsed = time.monotonic() - started
        if not isinstance(result, dict) or not isinstance(result.get("metadata", {}), dict):
            result = failure("Malformed worker response")
        if result.get("metadata", {}).get("resource_usage_unknown"):
            result["metadata"].pop("device_lease_receipt", None)
            return result
        return stamp_device_lease(result, lease, elapsed)
    return wrapped


def lease_elapsed(lease, result):
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    receipt = metadata.get("device_lease_receipt", {}) if isinstance(metadata, dict) else {}
    if not isinstance(receipt, dict):
        receipt = {}
    valid = (receipt.get("protocol") == LEASE_PROTOCOL and receipt.get("nonce") == lease["nonce"]
             and receipt.get("reserved_seconds") == lease["reserved_seconds"]
             and receipt.get("method") == "trusted_parent_monotonic_wall"
             and receipt.get("receipt_sha256") == fingerprint({key: value for key, value in receipt.items() if key != "receipt_sha256"}))
    if lease.get("gpu_busy") is not None:
        valid = valid and receipt.get("lease_sha256") == fingerprint(lease)
    return receipt.get("elapsed_seconds") if valid else None


def settle_device_lease(ledger, lease, result):
    return ledger.settle(lease["nonce"], lease_elapsed(lease, result))


def gpu_charge(lease, result):
    """Keep observed GPU time distinct from a conservative quota charge."""
    elapsed = lease_elapsed(lease, result)
    proof = {"lease": lease, "parent_elapsed_seconds": elapsed, "observed_gpu_busy_seconds": None,
             "charge_upper_bound_seconds": elapsed, "method": "confirmed_worker_wall_upper_bound",
             "raw_sha256": None, "scope": "verified_worker_process_CUDA; other_processes_unobserved; model_inference_and_final_excluded"}
    if not amount(elapsed) or elapsed > lease["reserved_seconds"]:
        proof.update(charge_upper_bound_seconds=None, method="usage_unknown_or_exceeded")
        return proof
    if not result.get("metadata", {}).get("origin_capture") or result["metadata"].get("tool_checks"):
        return proof
    try:
        from ..analysis.performance import interval_union
        from ..provenance.origin_task import scored_origin, validate_scored

        request, _ = validate_scored(result)
        if (request["eval_config"].get("device_lease") != lease or request.get("task", "evaluate") != "evaluate"
                or request["eval_config"].get("use_hidden_tests") or result.get("metadata", {}).get("tool_checks")):
            raise ValueError("Actual activity does not cover this complete development request")
        summary = scored_origin(result)
        activities = result["metadata"]["origin_capture"]["raw"]["activities"]
        if any(row["device"] != 0 for row in activities):
            raise ValueError("Development quota covers one CUDA device")
        busy = interval_union([(row["start_ns"], row["end_ns"]) for row in activities]) / 1e9
        if not amount(busy) or busy > elapsed:
            raise ValueError("Activity union exceeds the confirmed parent envelope")
        proof.update(charge_upper_bound_seconds=busy, observed_gpu_busy_seconds=busy,
                     method="actual_worker_continuous_CUPTI_union", raw_sha256=summary["raw_sha256"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        pass  # The independently confirmed wall envelope still bounds this quota.
    return proof


def settle_gpu_lease(ledger, lease, result, *, index, task):
    proof = {**gpu_charge(lease, result), "round": index, "task": task}
    compliant = ledger.settle(lease["gpu_busy"]["nonce"], proof["charge_upper_bound_seconds"])
    ledger.entries[-1]["basis"] = proof
    return compliant


def validate_gpu_artifacts(config, pid, sid, snapshot):
    from ..common import read_json

    seen = set()
    for entry in snapshot["entries"]:
        basis = entry.get("basis", {})
        index, task, lease = basis.get("round"), basis.get("task"), basis.get("lease", {})
        if (type(index) is not int or not 1 <= index <= config.optimization_rounds or task not in {"evaluate", "profile"}
                or (index, task) in seen or lease.get("gpu_busy", {}).get("nonce") != entry["nonce"]
                or lease["gpu_busy"].get("limit_seconds") != config.agent_gpu_busy_seconds
                or lease.get("reserved_seconds") != entry["reserved"]):
            return False
        seen.add((index, task))
        directory = config.run_dir / "work" / f"p{pid}_s{sid}" / "rounds" / f"{index:03}"
        value = read_json(directory / "standard.json", {}) if task == "evaluate" else read_json(directory / "feedback.json", {}).get("profile", {})
        validate_device_lease({"eval_config": {"device_lease": lease, "timeout": lease.get("execution_timeout_seconds")}})
        expected = {**gpu_charge(lease, value), "round": index, "task": task}
        if basis != expected or entry["observed"] != expected["charge_upper_bound_seconds"]:
            return False
    return True


def gpu_artifact_hashes(run_dir, pid, sid, snapshot):
    from pathlib import Path

    from ..common import read_json

    if type(pid) is not int or pid < 1 or type(sid) is not int or sid < 0:
        raise ValueError("Invalid GPU quota sample identity")
    root, output = Path(run_dir), {}
    for entry in snapshot["entries"]:
        basis = entry["basis"]
        index, task = basis["round"], basis["task"]
        if type(index) is not int or index < 1 or task not in {"evaluate", "profile"}:
            raise ValueError("Invalid GPU quota request identity")
        name = f"work/p{pid}_s{sid}/rounds/{index:03}/" + ("standard.json" if task == "evaluate" else "feedback.json")
        path = root / name
        if name in output or not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("GPU quota artifact is missing, duplicated or outside its run")
        output[name] = fingerprint(read_json(path))
    return output


def active(config):
    return bool(config.agent_token_limit or config.agent_output_token_limit or config.agent_device_seconds or config.agent_gpu_busy_seconds)


def score_issue(config, pid, sid):
    if not active(config):
        return None
    if config.generation == "harbor":
        from ..runtime.harbor_budget import score_issue as harbor_issue

        return harbor_issue(config, pid, sid)
    from ..analysis.experiment import existing_budget, protocol
    from ..common import read_json
    from .generation import generation_signature

    budget = existing_budget(config)
    if budget.get("provenance_valid") is not True:
        return "resource_budget_generation_provenance_missing"
    recorded_history = budget.get("attempt_history", {})
    if not isinstance(recorded_history, dict) or not isinstance(recorded_history.get("slots"), dict):
        return "resource_budget_history_missing"
    attempts = recorded_history["slots"].get(f"{pid}_{sid}", [])
    if not isinstance(attempts, list) or len(attempts) != 1 or not isinstance(attempts[0], dict):
        return "resource_budget_requires_one_auditable_attempt"
    receipt = attempts[0].get("budget") or {}
    if not isinstance(receipt, dict):
        return "resource_budget_receipt_missing"
    count_fields = ["model_calls"] + (["development_evaluations", "profile_requests"] if config.agent_device_seconds or config.agent_gpu_busy_seconds else [])
    if any(not amount(receipt.get(name), integer=True) for name in count_fields):
        return "resource_budget_call_counts_missing"
    history = read_json(config.run_dir / "generation_attempts.json", {})
    sample_protocol = {**protocol(config), "samples_per_problem": 1}
    if (budget.get("attempt_history") != history or history.get("historical_attempts_unknown")
            or receipt.get("protocol") != sample_protocol or budget.get("protocol") != protocol(config)
            or budget.get("slots", {}).get(f"{pid}_{sid}") != receipt):
        return "resource_budget_protocol_or_history_mismatch"
    limits = {"total_tokens": (config.agent_token_limit, "tokens"),
              "output_tokens": (config.agent_output_token_limit, "tokens"),
              "device_lease": (config.agent_device_seconds, "device_lease_seconds"),
              "gpu_busy_quota": (config.agent_gpu_busy_seconds, "gpu_busy_upper_bound_seconds")}
    records = receipt.get("resource_budgets", {})
    if not isinstance(records, dict):
        return "resource_budget_receipt_missing"
    for name, (limit, unit) in limits.items():
        if limit and not validate_snapshot(records.get(name), limit, unit):
            return "resource_budget_missing_unknown_or_exceeded"
    # The current snapshot is also tied to the actual completed generation summary.
    summary = read_json(config.run_dir / "generation_summary.json", {}).get(f"{pid}_{sid}", {})
    if config.generation == "modal":
        from ..runtime.modal_backend import validate_generation

        try:
            validate_generation(summary, config.run_dir / "work" / f"p{pid}_s{sid}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return "resource_budget_modal_generation_missing_or_invalid"
    if (summary.get("budget") != receipt or summary.get("status") != "success" or attempts[0].get("status") != "success"
            or summary.get("signature") != attempts[0].get("signature")
            or summary.get("signature") != generation_signature(config, pid, sid)):
        return "resource_budget_attempt_or_summary_mismatch"
    for name in ("total_tokens", "output_tokens"):
        if limits[name][0] and (records[name]["charged"] != receipt.get(name)
                              or len(records[name]["entries"]) != receipt.get("model_calls")):
            return "resource_budget_usage_mismatch"
    if config.agent_device_seconds and len(records["device_lease"]["entries"]) != (
            receipt.get("development_evaluations", -1) + receipt.get("profile_requests", -1)):
        return "resource_budget_device_request_mismatch"
    if config.agent_gpu_busy_seconds:
        if len(records["gpu_busy_quota"]["entries"]) != receipt["development_evaluations"] + receipt["profile_requests"]:
            return "resource_budget_gpu_request_mismatch"
        measured = [row.get("basis", {}).get("observed_gpu_busy_seconds") for row in records["gpu_busy_quota"]["entries"]]
        observed = sum(measured) if all(amount(value) for value in measured) else None
        if (receipt.get("gpu_busy_charge_upper_bound_seconds") != records["gpu_busy_quota"]["charged"]
                or receipt.get("gpu_execution_seconds") != observed):
            return "resource_budget_gpu_usage_mismatch"
        try:
            gpu_artifact_hashes(config.run_dir, pid, sid, records["gpu_busy_quota"])
            if not validate_gpu_artifacts(config, pid, sid, records["gpu_busy_quota"]):
                return "resource_budget_gpu_evidence_changed_or_missing"
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return "resource_budget_gpu_evidence_changed_or_missing"
    if config.agent_backend == "local-transformers":
        from .local_model import checkpoint_digest

        if receipt.get("local_checkpoint_sha256") != [checkpoint_digest(config.model)]:
            return "resource_budget_local_checkpoint_mismatch"
    return None

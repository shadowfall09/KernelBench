"""Frozen task-specific ordinary-state semantics; no device SDK imports."""

from __future__ import annotations

import hashlib

PROTOCOL = "kernelbench_pro_task_state_policy_v1"


def plan(level, pid, source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if level in {1, 2, 3}:
        from ..numerics.operator_input_policy import plan as inputs

        if inputs(level, pid, source, 256)["reference_sha256"] != digest:
            raise ValueError("State policy reference differs from its reviewed task")
    elif level == 5:
        from ..benchmarks.suites import load_manifest

        row = next((row for row in load_manifest("production41")["problems"] if row["problem_id"] == pid), None)
        if row is None or row["reference_sha256"] != digest:
            raise ValueError("State policy reference differs from its production contract")
    else:
        raise ValueError("No reviewed task-state policy for this level")
    ordinary = {}
    if level == 3 and pid == 33:
        ordinary["@hidden"] = {"role": "persistent", "replacement": "reference_matched",
                               "initial_placement": "cpu_or_input_device",
                               "reason": "forward copies the explicit initial state then replaces hidden with the recurrent result"}
    elif level == 3 and pid == 41:
        ordinary["@h0"] = {"role": "unused_initialization", "replacement": "not_a_forward_state",
                           "initial_placement": "reference",
                           "reason": "forward passes its h0 argument to GRU and never reads self.h0"}
    return {"protocol": PROTOCOL, "reference_sha256": digest, "ordinary": ordinary,
            "registered": "all_parameters_and_buffers", "modes": [True, True, False, True, False],
            "parameter_updates": "two_nonzero_updates_in_each_mode", "unknown_ordinary_state": "reject"}


def validate_inline(payload):
    settings = payload["eval_config"]
    policy = settings.get("state_policy")
    if policy is not None or settings.get("state_validation") and {"level", "problem_id"} <= payload.keys():
        if policy != plan(payload["level"], payload["problem_id"], payload["ref_arch_src"]):
            raise ValueError("Missing or stale source-bound task-state policy")
    return policy

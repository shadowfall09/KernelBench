"""Replay source -> binary -> function -> launch -> activity origin diagnostics.

Receipts are diagnostic. This module does not set evidence.kernel_origin, and a
JSON declaration, package path or kernel name never establishes authorship.
"""

from __future__ import annotations

import math
import re

from .common import fingerprint
from .performance import interval_union

PROTOCOL = "kernelbench_pro_cuda_origin_v1"
MAX_RECORDS = 200_000
KINDS = {"kernel", "memcpy", "memset"}


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _index(rows, label):
    if not isinstance(rows, list) or len(rows) > MAX_RECORDS:
        raise ValueError(f"Invalid or excessive {label} records")
    indexed = {}
    for row in rows:
        key = row.get("id") if isinstance(row, dict) else None
        if not _integer(key, positive=True) or key in indexed:
            raise ValueError(f"Invalid or duplicated {label} identity")
        indexed[key] = row
    return indexed


def summarize(raw, *, minimum_custom_fraction=0.95):
    """Conservative exclusive custom busy-time fraction for every captured call.

    Graph replays currently have no supported node -> function collector, so they
    stay unknown. Copy/set traffic has no source-derived arithmetic identity and
    receives no custom credit. A complete capture with unknown activity is still
    incomplete attribution. All arithmetic uses interval unions, not summed
    overlapping durations.
    """
    if (type(minimum_custom_fraction) not in (int, float) or not math.isfinite(minimum_custom_fraction)
            or not 0 < minimum_custom_fraction <= 1):
        raise ValueError("Custom fraction must be finite and in (0, 1]")
    if not isinstance(raw, dict) or raw.get("protocol") != PROTOCOL:
        raise ValueError("Unknown origin capture protocol")
    modules = _index(raw.get("modules"), "module")
    functions = _index(raw.get("functions"), "function")
    proofs = _index(raw.get("proofs"), "compiler proof")
    windows = _index(raw.get("windows"), "call window")
    if not windows:
        raise ValueError("No candidate calls were captured")
    for row in modules.values():
        if (not _sha(row.get("binary_sha256")) or not _integer(row.get("binary_bytes"), positive=True)
                or not _integer(row.get("handle"), positive=True) or not _integer(row.get("context"), positive=True)):
            raise ValueError("Missing observed module bytes or handle")
    for row in functions.values():
        if (row.get("module") not in modules or not _integer(row.get("handle"), positive=True)
                or row.get("context") != modules[row["module"]]["context"]):
            raise ValueError("Function is not bound to an observed module in its context")
    proven = {}
    for row in proofs.values():
        fn = functions.get(row.get("function"), {})
        mod = modules.get(fn.get("module"), {})
        if (row.get("compiler") != "triton-3.7.0" or not _sha(row.get("source_sha256"))
                or not _sha(row.get("jit_source_sha256")) or not _sha(row.get("compiler_metadata_sha256"))
                or row.get("source_sha256") != raw.get("candidate_sha256")
                or not mod or row.get("binary_sha256") != mod.get("binary_sha256")):
            raise ValueError("Compiler proof does not match frozen source and actual loaded bytes")
        if row["function"] in proven:
            raise ValueError("Ambiguous compiler ownership for a function generation")
        proven[row["function"]] = row
    launches = {}
    for row in raw.get("launches", []):
        if (not isinstance(row, dict) or not _integer(row.get("context_id"), positive=True)
                or not _integer(row.get("correlation"), positive=True)):
            raise ValueError("Invalid driver launch identity")
        key = (row["context_id"], row["correlation"])
        if key in launches:
            raise ValueError("Ambiguous driver launch correlation")
        launches[key] = row
    activities = raw.get("activities")
    if not isinstance(activities, list) or len(activities) > MAX_RECORDS or len(launches) > MAX_RECORDS:
        raise ValueError("Missing or excessive activity/launch records")
    for row in activities:
        if (not isinstance(row, dict) or row.get("kind") not in KINDS
                or any(not _integer(row.get(key)) for key in ("start_ns", "end_ns", "device", "stream", "context_id", "correlation", "graph_id", "graph_node_id"))
                or row["end_ns"] <= row["start_ns"]):
            raise ValueError("Invalid device activity interval or identity")
    ordered = sorted(windows.values(), key=lambda row: row.get("start_ns", -1))
    for idx, row in enumerate(ordered):
        if (any(not _integer(row.get(key)) for key in ("start_ns", "end_ns", "device", "seed"))
                or row["end_ns"] <= row["start_ns"] or not isinstance(row.get("input_signature"), list)
                or row.get("no_grad") is not True or type(row.get("training")) is not bool
                or row.get("output_verified") is not True
                or (idx and ordered[idx - 1]["end_ns"] > row["start_ns"])):
            raise ValueError("Invalid, overlapping, unverified or mode-unbound candidate call")
    capture_ok = (raw.get("capture_complete") is True and raw.get("source_intact") is True
                  and raw.get("hooks_intact") is True and raw.get("dropped_records") == 0
                  and type(raw.get("dropped_records")) is int and raw.get("errors") == []
                  and _sha(raw.get("candidate_sha256")) and raw.get("simulated") is False)
    calls = []
    for window in ordered:
        rows = []
        for activity in activities:
            if activity["device"] != window["device"]:
                continue
            start = max(window["start_ns"], activity["start_ns"])
            end = min(window["end_ns"], activity["end_ns"])
            if end <= start:
                continue
            launch = launches.get((activity["context_id"], activity["correlation"]), {})
            origin = "unknown"
            proof = proven.get(launch.get("function"))
            if (activity["kind"] == "kernel" and not activity["graph_id"] and not activity["graph_node_id"]
                    and launch.get("call") == window["id"] and proof is not None
                    and launch.get("exit_success") is True):
                origin = "custom"
            rows.append({**activity, "start_ns": start, "end_ns": end, "origin": origin})
        busy = interval_union([(row["start_ns"], row["end_ns"]) for row in rows])
        other = interval_union([(row["start_ns"], row["end_ns"]) for row in rows if row["origin"] != "custom"])
        custom_exclusive = max(0, busy - other)
        unknowns = sum(row["origin"] == "unknown" for row in rows)
        complete = capture_ok and bool(rows) and not unknowns
        share = custom_exclusive / busy if busy else None
        status = "not_measured" if not complete else ("passed" if share >= minimum_custom_fraction else "failed")
        # Even a conservative lower bound below the policy is inconclusive when
        # some attribution is unknown; it must not be presented as a measured fail.
        calls.append({"id": window["id"], "seed": window["seed"], "status": status,
                      "device_busy_ns": busy, "custom_exclusive_ns": custom_exclusive,
                      "custom_exclusive_fraction_lower_bound": share, "unknown_activity_count": unknowns,
                      "activity_count": len(rows), "kernel_count": sum(row["kind"] == "kernel" for row in rows),
                      "activities": rows})
    overall = "failed" if any(row["status"] == "failed" for row in calls) else (
        "passed" if all(row["status"] == "passed" for row in calls) else "not_measured")
    return {"protocol": PROTOCOL, "raw_sha256": fingerprint(raw), "status": overall,
            "minimum_custom_fraction": minimum_custom_fraction, "calls": calls,
            "timings_are_diagnostic": True, "formal_qualification_implemented": False,
            "library_and_framework_attribution_implemented": False,
            "output_causal_use_proven": False, "uninstrumented_path_equivalence_proven": False,
            "security_isolation_certified": False}

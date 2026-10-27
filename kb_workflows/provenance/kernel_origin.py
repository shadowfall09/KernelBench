"""Replay source -> binary -> function -> launch -> activity origin diagnostics.

Receipts are diagnostic. This module does not set evidence.kernel_origin, and a
JSON declaration, package path or kernel name never establishes authorship.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left

from ..analysis.performance import interval_union
from ..common import fingerprint
from .cpp_origin import Extensions
from .origin_binaries import binary_owners
from .origin_graphs import validate_updates
from .triton_origin import EMISSION_PROTOCOL, Frontend

PROTOCOL = "kernelbench_pro_cuda_origin_v4"
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


def _launch_index(rows):
    if not isinstance(rows, list) or len(rows) > MAX_RECORDS:
        raise ValueError("Invalid or excessive launch records")
    indexed = {}
    aliases = {}
    for row in rows:
        if (not isinstance(row, dict) or not _integer(row.get("context_id"), positive=True)
                or not _integer(row.get("correlation"), positive=True)
                or (row.get("runtime_correlation") is not None and not _integer(row["runtime_correlation"], positive=True))):
            raise ValueError("Invalid driver launch identity")
        key = (row["context_id"], row["correlation"])
        if key in indexed:
            raise ValueError("Ambiguous driver launch correlation")
        indexed[key] = row
        if row.get("runtime_correlation") is not None:
            alias = (row["context_id"], row["runtime_correlation"])
            aliases.setdefault(alias, []).append(row)
    for key, values in aliases.items():
        if len(values) == 1 and (key not in indexed or indexed[key] is values[0]):
            indexed[key] = values[0]
        else:
            # A runtime wrapper may launch several functions. It does not give
            # any one function exclusive ownership of its correlation ID.
            indexed.pop(key, None)
    return indexed


def summarize(raw, *, minimum_custom_fraction=0.95):
    """Conservative exclusive custom busy-time fraction for every captured call.

    Direct and graph calls require observed API/node provenance. Copy/set traffic
    receives runtime credit only from a matching successful native API or node,
    never custom credit. Unknown activity keeps attribution incomplete. All
    arithmetic uses interval unions, not summed overlapping durations.
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
        if (not _sha(row.get("source_sha256")) or row.get("source_sha256") != raw.get("candidate_sha256")
                or not mod or row.get("binary_sha256") != mod.get("binary_sha256")):
            raise ValueError("Compiler proof does not match frozen source and actual loaded bytes")
        if row.get("compiler") == "triton-3.7.0":
            emission = row.get("emission", {})
            frontend = row.get("frontend", {})
            if (not isinstance(emission, dict) or not isinstance(frontend, dict)
                    or not _sha(row.get("jit_source_sha256")) or not _sha(row.get("compiler_metadata_sha256"))
                    or emission.get("protocol") != EMISSION_PROTOCOL or emission.get("backend") != "cuda"
                    or emission.get("binary_sha256") != row["binary_sha256"]
                    or emission.get("metadata_sha256") != row["compiler_metadata_sha256"]
                    or emission.get("frontend_receipt_sha256") != frontend.get("receipt_sha256")
                    or not Frontend.validate(row.get("frontend"), source_sha256=row["source_sha256"], jit_source_sha256=row["jit_source_sha256"])):
                raise ValueError("Compiler proof has no complete frozen frontend dependency/AST receipt")
        elif row.get("compiler") == "torch-cpp-extension-2.12.0":
            if not Extensions.validate(row.get("cpp_build"), candidate_sha256=raw["candidate_sha256"], binary_sha256=mod["binary_sha256"]):
                raise ValueError("CUDA extension has no complete source/object compilation receipt")
        else:
            raise ValueError("Compiler proof uses an unsupported adapter")
        if row["function"] in proven:
            raise ValueError("Ambiguous compiler ownership for a function generation")
        proven[row["function"]] = row
    launches = _launch_index(raw.get("launches", []))
    graph_nodes = _index(raw.get("graph_nodes", []), "graph node")
    graph_updates = _index(raw.get("graph_updates", []), "graph update")
    graph_launches = _launch_index(raw.get("graph_launches", []))
    memory_calls = _launch_index(raw.get("memory_calls", []))
    for node in graph_nodes.values():
        if (not _integer(node.get("node_id"), positive=True) or not _integer(node.get("graph_id"), positive=True)
                or node.get("kind") not in {"kernel", "memcpy", "memset", "other"}
                or node.get("creation") not in {"driver_kernel", "driver_memory", "clone", "template_kernel_update", "exec_kernel_update"}
                or type(node.get("creation_success")) is not bool
                or (node.get("function") is not None and (node["function"] not in functions or node["kind"] != "kernel"))
                or (node.get("creation") == "clone" and node.get("original") is None and node.get("function") is not None)):
            raise ValueError("Invalid graph node/function provenance")
        if node.get("original") is not None:
            original = graph_nodes.get(node["original"])
            if (original is None or original["id"] >= node["id"] or node.get("creation") != "clone"
                    or original.get("function") != node.get("function")
                    or original.get("kind") != node.get("kind")
                    or (node["creation_success"] and original.get("creation_success") is not True)):
                raise ValueError("Graph clone does not retain a valid earlier function generation")
    for launch in graph_launches.values():
        if (not isinstance(launch.get("nodes"), dict) or len(launch["nodes"]) > MAX_RECORDS
                or not isinstance(launch.get("graph_ids"), list)
                or any(not _integer(value, positive=True) for value in launch["graph_ids"])):
            raise ValueError("Invalid executable graph snapshot")
        for uid, record in launch["nodes"].items():
            if (not isinstance(uid, str) or not uid.isdecimal() or str(int(uid)) != uid
                    or record not in graph_nodes or graph_nodes[record]["node_id"] != int(uid)):
                raise ValueError("Graph launch snapshot has a mismatched node generation")
    validate_updates(graph_updates, graph_nodes, functions, raw.get("graph_launches", []), limit=MAX_RECORDS)
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
                or type(row.get("training_after")) is not bool or row["training_after"] != row["training"]
                or row.get("output_verified") is not True
                or (idx and ordered[idx - 1]["end_ns"] > row["start_ns"])):
            raise ValueError("Invalid, overlapping, unverified or mode-unbound candidate call")
    capture_ok = (raw.get("capture_complete") is True and raw.get("source_intact") is True
                  and raw.get("hooks_intact") is True and raw.get("dropped_records") == 0
                  and type(raw.get("dropped_records")) is int and raw.get("errors") == []
                  and _sha(raw.get("candidate_sha256")) and raw.get("simulated") is False)
    library_owners = binary_owners(raw.get("binary_inventory"))
    # Native hidden portfolios can have thousands of windows. Join once, while
    # retaining every cross-window overlap, instead of scanning all events for
    # every call. Refuse excessive overlap expansion rather than truncate it.
    starts = [window["start_ns"] for window in ordered]
    per_window = {window["id"]: [] for window in ordered}
    associations = 0
    for activity in activities:
        index = bisect_left(starts, activity["end_ns"]) - 1
        while index >= 0 and ordered[index]["end_ns"] > activity["start_ns"]:
            window = ordered[index]
            if window["device"] == activity["device"]:
                per_window[window["id"]].append(activity)
                associations += 1
                if associations > MAX_RECORDS:
                    raise ValueError("Activity/window overlap limit exceeded")
            index -= 1
    calls = []
    for window in ordered:
        rows = []
        for activity in per_window[window["id"]]:
            start = max(window["start_ns"], activity["start_ns"])
            end = min(window["end_ns"], activity["end_ns"])
            if end <= start:
                continue
            launch = launches.get((activity["context_id"], activity["correlation"]), {})
            origin = "unknown"
            function_id = launch.get("function")
            node = {}
            if activity["graph_id"] or activity["graph_node_id"]:
                launch = graph_launches.get((activity["context_id"], activity["correlation"]), {})
                node = graph_nodes.get(launch.get("nodes", {}).get(str(activity["graph_node_id"])), {})
                function_id = (node.get("function") if activity["graph_id"] in launch.get("graph_ids", [])
                               and node.get("kind") == "kernel" and node.get("creation_success") is True else None)
            elif activity["kind"] in {"memcpy", "memset"}:
                launch = memory_calls.get((activity["context_id"], activity["correlation"]), {})
            proof = proven.get(function_id)
            function = functions.get(function_id, {})
            module = modules.get(function.get("module"), {})
            if (activity["kind"] == "kernel" and launch.get("call") == window["id"] and launch.get("exit_success") is True):
                # Frozen installed-image ownership takes precedence over a
                # copied candidate root emitting identical library bytes.
                origin = library_owners.get(module.get("binary_sha256"), "custom" if proof else "unknown")
            elif (activity["kind"] in {"memcpy", "memset"} and launch.get("call") == window["id"]
                    and launch.get("exit_success") is True):
                if activity["graph_id"] or activity["graph_node_id"]:
                    memory_ok = (activity["graph_id"] in launch.get("graph_ids", [])
                                 and node.get("kind") == activity["kind"] and node.get("creation_success") is True)
                else:
                    memory_ok = launch.get("kind") == activity["kind"]
                if memory_ok:
                    origin = "runtime"
            # A binary may contain several entrypoints with different work.
            # Keep resolved identity/configuration separate from origin credit.
            configuration = activity.get("configuration")
            identity = None
            if (activity["kind"] == "kernel" and origin in {"custom", "library", "framework"}
                    and isinstance(function.get("symbol"), str) and function["symbol"]
                    and activity.get("name") == function["symbol"] and _sha(module.get("binary_sha256"))
                    and isinstance(configuration, dict)
                    and set(configuration) == {"grid", "block", "dynamic_shared_bytes", "static_shared_bytes"}
                    and all(isinstance(configuration.get(key), list) and len(configuration[key]) == 3
                            and all(_integer(value, positive=True) for value in configuration[key]) for key in ("grid", "block"))
                    and all(_integer(configuration[key]) for key in ("dynamic_shared_bytes", "static_shared_bytes"))):
                identity = {"binary_sha256": module["binary_sha256"], "symbol": function["symbol"],
                            "configuration": configuration}
            rows.append({**activity, "start_ns": start, "end_ns": end, "origin": origin, "kernel_identity": identity,
                         **({"memory_interval_complete": start == activity["start_ns"] and end == activity["end_ns"]}
                            if activity["kind"] != "kernel" else {})})
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
                      "track_verdicts": {"custom": status, "library": "passed" if complete else "not_measured"},
                      "device_busy_ns": busy, "custom_exclusive_ns": custom_exclusive,
                      "custom_exclusive_fraction_lower_bound": share, "unknown_activity_count": unknowns,
                      "activity_count": len(rows), "kernel_count": sum(row["kind"] == "kernel" for row in rows),
                      "memory_activity": memory_traffic({"activities": rows}, complete=(
                          raw.get("capture_complete") is True and raw.get("source_intact") is True
                          and raw.get("hooks_intact") is True and _integer(raw.get("dropped_records"))
                          and raw["dropped_records"] == 0 and raw.get("errors") == [])),
                      "by_origin_union_ns": {kind: interval_union([(row["start_ns"], row["end_ns"]) for row in rows
                                                                  if row["origin"] == kind])
                                             for kind in ("custom", "library", "framework", "runtime", "unknown")},
                      "activities": rows})
    overall = "failed" if any(row["status"] == "failed" for row in calls) else (
        "passed" if all(row["status"] == "passed" for row in calls) else "not_measured")
    return {"protocol": PROTOCOL, "raw_sha256": fingerprint(raw), "status": overall,
            "track_verdicts": {"custom": overall, "library": "passed" if all(row["track_verdicts"]["library"] == "passed" for row in calls) else "not_measured"},
            "minimum_custom_fraction": minimum_custom_fraction, "calls": calls,
            "timings_are_diagnostic": True, "formal_qualification_implemented": False,
            "library_and_framework_attribution_implemented": True,
            "graph_attribution_implemented": True,
            "output_causal_use_proven": False, "uninstrumented_path_equivalence_proven": False,
            "security_isolation_certified": False}


def memory_traffic(call, *, complete=True):
    """Logical transfer payload is distinct from measured DRAM/cache traffic."""
    rows = [row for row in call["activities"] if row["kind"] in {"memcpy", "memset"}]
    kinds = {"host": {1, 2}, "device": {3, 6}, "array": {4}}
    directions = {1: ("host", "device"), 2: ("device", "host"), 3: ("host", "array"),
                  4: ("array", "host"), 5: ("array", "array"), 6: ("array", "device"),
                  7: ("device", "array"), 8: ("device", "device"), 9: ("host", "host")}
    payload, reads, writes, unknown_payload, unknown_direction = 0, 0, 0, 0, 0
    by_kind = {kind: {"operations": 0, "payload_bytes": 0} for kind in ("memcpy", "memset")}
    for row in rows:
        kind, size = row["kind"], row.get("bytes")
        by_kind[kind]["operations"] += 1
        if not _integer(size) or size >= 2**64 or row.get("memory_interval_complete") is not True:
            unknown_payload += 1
            unknown_direction += 1
            by_kind[kind]["payload_bytes"] = None
            continue
        payload += size
        if by_kind[kind]["payload_bytes"] is not None:
            by_kind[kind]["payload_bytes"] += size
        if kind == "memset":
            if type(row.get("memory_kind")) is int and row["memory_kind"] in kinds["device"] | kinds["array"]:
                writes += size
                continue
        else:
            source, destination = directions.get(row.get("copy_kind") if type(row.get("copy_kind")) is int else None, (None, None))
            if (source is not None and type(row.get("src_kind")) is int and type(row.get("dst_kind")) is int
                    and row["src_kind"] in kinds[source] and row["dst_kind"] in kinds[destination]):
                reads += size * (source != "host")
                writes += size * (destination != "host")
                continue
        unknown_direction += 1
    if not complete:
        for item in by_kind.values():
            item["payload_bytes"] = None
    return {"operations": len(rows), "by_kind": by_kind,
            "capture_complete": complete,
            "payload_bytes": payload if complete and not unknown_payload else None,
            "logical_device_read_bytes": reads if complete and not unknown_direction else None,
            "logical_device_write_bytes": writes if complete and not unknown_direction else None,
            "unknown_payload_operations": unknown_payload, "unknown_direction_operations": unknown_direction,
            "basis": "whole_activity_logical_payload; not_DRAM_cache_counters_or_execution_SOL"}


def workload_identity(call):
    """A multiset preserves concurrent-launch multiplicity without timing order."""
    activities = call.get("activities", [])
    if (not activities or any(row.get("kind") != "kernel" or not isinstance(row.get("kernel_identity"), dict)
                              for row in activities)):
        raise ValueError("Complete kernel identity/configuration is required; native memory work is not covered")
    return {"kernel_count": len(activities), "kernel_sha256": sorted(fingerprint(row["kernel_identity"]) for row in activities)}

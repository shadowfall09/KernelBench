"""Offline activity-timeline analysis; unknown origins and counters stay unknown."""

from __future__ import annotations

import math

from .calibration import positive


def interval_union(intervals):
    ordered = []
    for start, end in intervals:
        if type(start) not in (int, float) or not math.isfinite(start) or start < 0 or not positive(end) or end <= start:
            raise ValueError("Activity intervals need finite nonnegative starts and later ends")
        ordered.append((start, end))
    merged = []
    for start, end in sorted(ordered):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def timeline(trace, *, device, marker="kernelbench_profile", origins=None):
    """Parse a single marked Chrome/Kineto capture, in microseconds.

    Origin classifications must come from a verifier-owned correlation map. Kernel
    names alone do not establish authorship. CUDA runtime calls are CPU events.
    """
    events = trace.get("traceEvents", [])
    markers = [row for row in events if row.get("name") == marker and row.get("ph") == "X"]
    if len(markers) != 1:
        raise ValueError("Trace requires exactly one measured wall interval")
    span = markers[0]
    start, end = span["ts"], span["ts"] + span["dur"]
    interval_union([(start, end)])
    if not trace.get("deviceProperties"):
        raise ValueError("No device metadata in trace")
    rows = []
    for event in events:
        category = event.get("cat", "")
        if event.get("ph") != "X" or category not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        args = event.get("args", {})
        # Kineto traces report the logical device ordinal in the event arguments.
        if args.get("device") != device:
            continue
        left, right = event["ts"], event["ts"] + event["dur"]
        interval_union([(left, right)])
        if right <= start or left >= end:
            continue
        correlation = args.get("correlation")
        origin = (origins or {}).get(str(correlation), "unknown")
        if origin not in {"custom", "library", "unknown"}:
            raise ValueError("Unknown origin classification")
        rows.append({"name": event.get("name"), "start_us": max(start, left), "end_us": min(end, right),
                     "origin": origin, "kind": category, "correlation": correlation})
    if not rows:
        raise ValueError("No measured device activity; absence is not zero cost")
    busy = interval_union([(row["start_us"], row["end_us"]) for row in rows])
    custom = interval_union([(row["start_us"], row["end_us"]) for row in rows if row["origin"] == "custom"])
    # Exclude overlap with any non-custom activity for a conservative custom share.
    other = interval_union([(row["start_us"], row["end_us"]) for row in rows if row["origin"] != "custom"])
    conservative_custom = max(0, busy - other)
    return {"protocol": "device_activity_union_v1", "wall_ms": (end - start) / 1000,
            "device_busy_ms": busy / 1000, "kernel_count": sum(row["kind"] == "kernel" for row in rows),
            "activity_count": len(rows), "custom_union_ms": custom / 1000,
            "custom_exclusive_fraction": conservative_custom / busy,
            "unknown_origin_count": sum(row["origin"] == "unknown" for row in rows),
            "origin_map_supplied": origins is not None, "activities": rows,
            "timings_are_diagnostic": True, "hardware_counters_collected": False,
            "scored_wall_times_replaced": False}


def decomposition(wall_ms, busy_ms, sol_algorithm_ms, *, sol_execution_ms=None, counters=None):
    if not all(map(positive, (wall_ms, busy_ms, sol_algorithm_ms))) or busy_ms > wall_ms * (1 + 1e-9):
        raise ValueError("Decomposition needs positive times and busy time no greater than wall")
    measured, issue, evidence_kind, hardware_counters = False, None, None, False
    if counters is not None:
        from ..hardware.neuron_work import MODEL as NEURON_MODEL
        from ..hardware.neuron_work import validate_model as validate_neuron_model
        from ..provenance.work_counters import validate_model

        try:
            neuron = isinstance(counters, dict) and counters.get("protocol") == NEURON_MODEL
            model = validate_neuron_model(counters) if neuron else validate_model(counters)
            if sol_execution_ms is not None and sol_execution_ms != model["sol_execution_ms"]:
                raise ValueError("Supplied execution SOL differs from its raw work counters")
            sol_execution_ms, measured = model["sol_execution_ms"], True
            evidence_kind = "reviewed_neuron_instruction_annotations" if neuron else "reviewed_Nsight_hardware_counters"
            hardware_counters = not neuron
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            issue = str(exc)
            if sol_execution_ms is not None:
                raise ValueError("Separate A/E factors require replayable source-bound work counters and calibration") from exc
    if sol_execution_ms is not None and (not measured or not positive(sol_execution_ms)):
        raise ValueError("Separate A/E factors require replayable source-bound work counters and calibration")
    algorithm = sol_execution_ms / sol_algorithm_ms if sol_execution_ms is not None else None
    efficiency = busy_ms / sol_execution_ms if sol_execution_ms is not None else None
    return {"A": algorithm, "E": efficiency, "AE": busy_ms / sol_algorithm_ms, "O": wall_ms / busy_ms,
            "sol_algorithm_ms": sol_algorithm_ms, "wall_ms": wall_ms, "device_busy_ms": busy_ms,
            "counters": counters, "counter_issue": issue,
            "model_consistent": sol_algorithm_ms <= busy_ms and (sol_execution_ms is None or
                sol_algorithm_ms <= sol_execution_ms <= busy_ms), "hardware_counters_collected": hardware_counters,
            "execution_model_replayed": measured, "work_evidence_kind": evidence_kind}


def native_timeline(report, *, allow_simulated=False):
    """Join replayed native calls to wall and activity intervals from one clock.

    CUPTI window duration includes the candidate invocation and completion
    barrier. It is an instrumented diagnostic, separate from native score time.
    """
    from ..common import fingerprint
    from ..provenance.origin_task import replay

    summary = replay(report, allow_simulated=allow_simulated)
    raw, coverage = report["raw"], report["native_coverage"]
    if (raw.get("capture_complete") is not True or raw.get("source_intact") is not True
            or raw.get("hooks_intact") is not True or raw.get("errors") != []
            or type(raw.get("dropped_records")) is not int or raw["dropped_records"] != 0):
        raise ValueError("Native timeline requires complete source-bound capture and drop accounting")
    calls, totals = [], {}
    native_times = {stage: {item["id"]: item["wall_ms"] for item in document["native_timings"]}
                    for stage, document in report["native_runs"].items()}
    for observed, window, activity in zip(report["native_runs"]["profile"]["calls"], raw["windows"], summary["calls"], strict=True):
        if observed["id"] != window["id"] or observed["id"] != activity["id"]:
            raise ValueError("Native timeline call/window/activity identities differ")
        wall = (window["end_ns"] - window["start_ns"]) / 1e6
        busy = activity["device_busy_ns"] / 1e6
        if not positive(wall) or busy < 0 or busy > wall * (1 + 1e-9):
            raise ValueError("Native activity exceeds its own wall interval")
        calls.append({"id": observed["id"], "phase": observed["phase"], "seed": observed["seed"],
                      "label": observed["label"], "order": observed["order"], "native_call_sha256": fingerprint(observed),
                      "status": ("simulated" if coverage["simulated"] else "measured") if busy else "not_measured",
                      "profile_window_wall_ms": wall, "device_busy_ms": busy if busy else None,
                      "native_control_wall_ms": native_times["control"].get(observed["id"]),
                      "native_profile_wall_ms": native_times["profile"].get(observed["id"]),
                      "uncovered_wall_ms": max(0, wall - busy) if busy else None, "O": wall / busy if busy else None,
                      "A": None, "E": None, "AE": None, "kernel_count": activity["kernel_count"],
                      "activity_count": activity["activity_count"], "unknown_activity_count": activity["unknown_activity_count"],
                      "origin_union_ms": {key: value / 1e6 for key, value in activity["by_origin_union_ns"].items()},
                      "custom_exclusive_fraction_lower_bound": activity["custom_exclusive_fraction_lower_bound"],
                      "origin_verdicts": activity["track_verdicts"]})
        group = totals.setdefault(observed["phase"], {"calls": 0, "profile_window_wall_ms": 0., "observed_device_busy_ms": 0.})
        group["calls"] += 1
        group["profile_window_wall_ms"] += wall
        group["observed_device_busy_ms"] += busy
    for phase, group in totals.items():
        complete = all(call["status"] != "not_measured" for call in calls if call["phase"] == phase)
        group["device_busy_ms"] = group["observed_device_busy_ms"] if complete else None
        group["O"] = group["profile_window_wall_ms"] / group["observed_device_busy_ms"] if complete else None
        group["complete"] = complete
    value = {"protocol": "kernelbench_pro_native_timeline_v1", "origin_receipt_sha256": report["receipt_sha256"],
             "status": ("simulated" if coverage["simulated"] else "measured") if all(row["complete"] for row in totals.values()) else "not_measured",
             "calls": calls, "by_phase": totals, "device_fingerprint": raw["binding"]["device_fingerprint"],
             "scope": "same_CUPTI_window_invocation_and_completion; input_output_hashing_and_reference_excluded",
             "counter_native_call_join_implemented": False, "formal_qualification_implemented": False,
             "hardware_counters_collected": False, "timings_are_diagnostic": True, "scored_wall_times_replaced": False}
    value["receipt_sha256"] = fingerprint(value)
    return value


def launch_only(reference, candidate, *, launch_ms, speedup_ci95, noise_ci95, relative_busy_tolerance=0.05):
    """A diagnostic category, requiring a noise-separated speedup and measured launch cost."""
    if (not positive(launch_ms) or not 0 <= relative_busy_tolerance < 1
            or len(speedup_ci95) != 2 or len(noise_ci95) != 2
            or not all(map(positive, speedup_ci95 + noise_ci95))):
        raise ValueError("Launch attribution requires calibration and valid paired intervals")
    if any(not positive(row.get(key)) for row in (reference, candidate) for key in ("wall_ms", "device_busy_ms")):
        raise ValueError("Launch attribution needs both measured timelines")
    if any(type(row.get("kernel_count")) is not int or row["kernel_count"] < 0 for row in (reference, candidate)):
        raise ValueError("Kernel counts must be nonnegative integers")
    if speedup_ci95[0] > speedup_ci95[1] or noise_ci95[0] > noise_ci95[1]:
        raise ValueError("Confidence interval endpoints must be ordered")
    saved_kernels = reference["kernel_count"] - candidate["kernel_count"]
    wall_saved = reference["wall_ms"] - candidate["wall_ms"]
    busy_change = abs(reference["device_busy_ms"] - candidate["device_busy_ms"]) / reference["device_busy_ms"]
    established = (speedup_ci95[0] > max(1, noise_ci95[1]) and saved_kernels > 0 and wall_saved > 0
                   and busy_change <= relative_busy_tolerance and wall_saved <= saved_kernels * launch_ms)
    return {"launch_only": established, "saved_kernel_count": saved_kernels,
            "wall_saved_ms": wall_saved, "estimated_launch_savings_ms": max(0, saved_kernels) * launch_ms,
            "relative_busy_change": busy_change, "relative_busy_tolerance": relative_busy_tolerance,
            "classification_is_diagnostic": True, "counter_evidence_substituted": False}

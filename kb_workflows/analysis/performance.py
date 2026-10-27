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


def native_timeline(report, *, counter_report=None, calibration=None, bound=None, allow_simulated=False):
    """Join replayed native calls to wall and activity intervals from one clock.

    CUPTI window duration includes the candidate invocation and completion
    barrier. It is an instrumented diagnostic, separate from native score time.
    """
    from ..common import fingerprint
    from ..provenance.origin_task import replay

    summary = replay(report, allow_simulated=allow_simulated)
    models = None
    if (calibration is None) != (bound is None) or calibration is not None and counter_report is None:
        raise ValueError("Native A/E requires counters, same-device calibration and reviewed algorithm certificates together")
    if counter_report is not None:
        from ..provenance.counter_task import execution_models, validate

        validate(counter_report, allow_simulated=allow_simulated)
        if counter_report["origin"]["receipt_sha256"] != report["receipt_sha256"]:
            raise ValueError("Native counters belong to a different source capture/request")
        if calibration is not None:
            models = execution_models(counter_report, calibration, allow_simulated=allow_simulated)["calls"]
    raw, coverage = report["raw"], report["native_coverage"]
    if (raw.get("capture_complete") is not True or raw.get("source_intact") is not True
            or raw.get("hooks_intact") is not True or raw.get("errors") != []
            or type(raw.get("dropped_records")) is not int or raw["dropped_records"] != 0):
        raise ValueError("Native timeline requires complete source-bound capture and drop accounting")
    calls, totals = [], {}
    certificates = {}
    if models is not None:
        for item in bound if isinstance(bound, list) else [bound]:
            if not isinstance(item, dict) or not isinstance(item.get("input_signature"), list):
                raise ValueError("Native algorithm certificate lacks its input geometry")
            key = fingerprint({"signature": item["input_signature"]})
            if key in certificates:
                raise ValueError("Each native geometry requires exactly one algorithm certificate")
            certificates[key] = item
    native_times = {stage: {item["id"]: item["wall_ms"] for item in document["native_timings"]}
                    for stage, document in report["native_runs"].items()}
    scored = report["native_request"].get("_scored_result")
    scored_times = ({item["id"]: item["wall_ms"] for item in scored["metadata"]["native_call_receipt"]["native_timings"]}
                    if scored is not None else {})
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
                      "native_scored_wall_ms": scored_times.get(observed["id"]),
                      "model_state_before_sha256": observed["model_state_before"]["sha256"],
                      "model_state_after_sha256": observed["model_state_after"]["sha256"],
                      "model_state_changed": observed["model_state_before"]["sha256"] != observed["model_state_after"]["sha256"],
                      "uncovered_wall_ms": max(0, wall - busy) if busy else None, "O": wall / busy if busy else None,
                      "A": None, "E": None, "AE": None, "kernel_count": activity["kernel_count"],
                      "activity_count": activity["activity_count"], "unknown_activity_count": activity["unknown_activity_count"],
                      "origin_union_ms": {key: value / 1e6 for key, value in activity["by_origin_union_ns"].items()},
                      "custom_exclusive_fraction_lower_bound": activity["custom_exclusive_fraction_lower_bound"],
                      "origin_verdicts": activity["track_verdicts"]})
        if models is not None:
            import hashlib

            from .headroom import certificate

            model = models[len(calls) - 1]
            if model["id"] != observed["id"] or model["native_call_sha256"] != fingerprint(observed):
                raise ValueError("Native execution model belongs to another call")
            selected = certificates.get(observed["input_signature_sha256"])
            if selected is None:
                raise ValueError("Each native call needs exactly one reviewed algorithm certificate for its geometry")
            metadata = {"device_fingerprint": raw["binding"]["device_fingerprint"],
                        "input_signature": window["input_signature"], "precision": report["native_request"]["eval_config"]["precision"]}
            checked = certificate(selected, calibration, metadata,
                                  source_sha256=hashlib.sha256(report["native_request"]["ref_arch_src"].encode()).hexdigest(),
                                  digest=report["native_request"]["eval_config"]["evaluator_sha256"])
            algorithm, execution = checked["sol_algorithm_ms"], model["sol_execution_ms"]
            calls[-1].update(sol_algorithm_ms=algorithm, sol_execution_ms=execution, A=execution / algorithm,
                             E=busy / execution if busy else None, AE=busy / algorithm if busy else None,
                             model_consistent=bool(busy and algorithm <= execution <= busy))
        group = totals.setdefault(observed["phase"], {"calls": 0, "profile_window_wall_ms": 0., "observed_device_busy_ms": 0.})
        group["calls"] += 1
        group["profile_window_wall_ms"] += wall
        group["observed_device_busy_ms"] += busy
    for phase, group in totals.items():
        complete = all(call["status"] != "not_measured" for call in calls if call["phase"] == phase)
        group["device_busy_ms"] = group["observed_device_busy_ms"] if complete else None
        group["O"] = group["profile_window_wall_ms"] / group["observed_device_busy_ms"] if complete else None
        group["complete"] = complete
        if models is not None:
            selected = [row for row in calls if row["phase"] == phase]
            algorithm = sum(row["sol_algorithm_ms"] for row in selected)
            execution = sum(row["sol_execution_ms"] for row in selected)
            group.update(sol_algorithm_ms=algorithm, sol_execution_ms=execution, A=execution / algorithm,
                         E=group["observed_device_busy_ms"] / execution if complete else None,
                         AE=group["observed_device_busy_ms"] / algorithm if complete else None)
    value = {"protocol": "kernelbench_pro_native_timeline_v1", "origin_receipt_sha256": report["receipt_sha256"],
             "status": ("simulated" if coverage["simulated"] else "measured") if all(row["complete"] for row in totals.values()) else "not_measured",
             "calls": calls, "by_phase": totals, "device_fingerprint": raw["binding"]["device_fingerprint"],
             "scope": "same_CUPTI_window_invocation_and_completion; input_output_hashing_and_reference_excluded",
             "counter_native_call_join_implemented": True, "counter_report_joined": counter_report is not None,
             "counter_receipt_sha256": counter_report["receipt_sha256"] if counter_report is not None else None,
             "calibration_sha256": calibration["receipt_sha256"] if models is not None else None,
             "bound_sha256": fingerprint({"certificates": bound}) if models is not None else None,
             "execution_model_replayed": models is not None, "formal_qualification_implemented": False,
             "hardware_counters_collected": counter_report is not None and counter_report["simulated"] is False,
             "timings_are_diagnostic": True, "scored_wall_times_replaced": False}
    value["receipt_sha256"] = fingerprint(value)
    return value


def launch_only(reference, candidate, *, launch_ms, speedup_ci95, noise_ci95, relative_busy_tolerance=0.05,
                launch_tolerance=0.2):
    """A diagnostic category, requiring a noise-separated speedup and measured launch cost."""
    if (not positive(launch_ms) or not 0 <= relative_busy_tolerance < 1 or not 0 <= launch_tolerance < 1
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
                   and busy_change <= relative_busy_tolerance
                   and abs(wall_saved - saved_kernels * launch_ms) <= saved_kernels * launch_ms * launch_tolerance)
    return {"launch_only": established, "saved_kernel_count": saved_kernels,
            "wall_saved_ms": wall_saved, "estimated_launch_savings_ms": max(0, saved_kernels) * launch_ms,
            "relative_busy_change": busy_change, "relative_busy_tolerance": relative_busy_tolerance,
            "launch_tolerance": launch_tolerance,
            "classification_is_diagnostic": True, "counter_evidence_substituted": False}


def scored_pairs(result):
    """Replay same-process reference/candidate/control windows, never a diagnostic run."""
    from ..common import fingerprint
    from ..evaluation.paired_task import call_plan
    from ..provenance.kernel_origin import _sha
    from ..provenance.origin_task import PAIRED_CAPTURE, scored_origin, validate_scored

    source = scored_origin(result)
    request, document = validate_scored(result)
    origin = result["metadata"]["origin_capture"]
    record = origin.get("paired_capture", {})
    if request.get("task") != "paired_timing" or record.get("protocol") != PAIRED_CAPTURE:
        raise ValueError("Actual paired device windows are missing")
    expected = [(index, seed, order, role) for index, seed, order in call_plan(
        trials=request["eval_config"].get("paired_trials", 30), warmup=3,
        seed=request["eval_config"].get("evaluation_seed", 42)) for role in order]
    rows, raw = record.get("calls"), origin["raw"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Paired device windows omit reference, candidate or control calls")
    candidates = {row["label"]: row for row in document["calls"]}
    windows = {row["id"]: row for row in raw["windows"]}
    times = {row["id"]: row["wall_ms"] for row in document["native_timings"]}
    pairs = result["metadata"]["paired_timing"]["pairs"]
    output, previous = [], -1
    for index, (row, (trial, seed, order, role)) in enumerate(zip(rows, expected, strict=True), 1):
        candidate = candidates[str(trial)]
        start, end, device = (row.get(key) for key in ("start_ns", "end_ns", "device"))
        if (row.get("id") != index or (row.get("trial"), row.get("seed"), row.get("order"), row.get("role")) != (trial, seed, order, role)
                or row.get("output_verified") is not True or type(device) is not int or device != 0
                or type(start) is not int or type(end) is not int or start < previous or end <= start
                or not positive(row.get("wall_ms")) or not _sha(row.get("output_content_sha256"))
                or row.get("input_signature_sha256") != candidate["input_signature_sha256"]):
            raise ValueError("Paired device window changed its schedule, output, input or timing")
        if role == "C":
            window = windows[candidate["id"]]
            if (row["wall_ms"] != times[candidate["id"]] or row["output_content_sha256"] != candidate["output_content_sha256"]
                    or not window["start_ns"] <= start < end <= window["end_ns"]):
                raise ValueError("Paired candidate window differs from its scored native call")
        if trial >= 3 and row["wall_ms"] != pairs[trial - 3][{"R": "reference_ms", "C": "candidate_ms", "B": "control_ms"}[role]]:
            raise ValueError("Paired device window differs from its raw performance sample")
        activities = [{**activity, "start_ns": max(start, activity["start_ns"]), "end_ns": min(end, activity["end_ns"])}
                      for activity in raw["activities"] if activity["device"] == device and activity["start_ns"] < end and activity["end_ns"] > start]
        if not activities:
            raise ValueError("Paired device call has no measured activity")
        busy = interval_union([(activity["start_ns"], activity["end_ns"]) for activity in activities]) / 1e6
        if busy > row["wall_ms"] * (1 + 1e-6):
            raise ValueError("Paired device activity exceeds its actual native wall sample")
        if trial >= 3:
            output.append({"trial": trial, "role": role, "wall_ms": row["wall_ms"], "device_busy_ms": busy,
                           "kernel_count": sum(activity["kind"] == "kernel" for activity in activities),
                           "activity_count": len(activities), "O": row["wall_ms"] / busy})
        previous = end
    return {"protocol": "kernelbench_pro_scored_paired_activity_v1", "calls": output,
            "origin_receipt_sha256": origin["receipt_sha256"], "result_sha256": fingerprint(result),
            "raw_sha256": source["raw_sha256"], "device_fingerprint": document["binding"]["device_fingerprint"],
            "timing_mode": source["native_timing_mode"], "scored_wall_times_replaced": False}


def paired_attribution(paired, calibration):
    """Classify every independent repeat from replayed actual device activity."""
    from ..common import evaluator_digest, fingerprint
    from .calibration import validate
    from .paired_statistics import paired_summary

    runs, bindings = paired.get("device_runs", []), paired["measurement_bindings"]
    if not runs or len(runs) != len(bindings):
        raise ValueError("Actual paired activity must cover every declared repeat")
    groups = []
    for repeat, (run, binding) in enumerate(zip(runs, bindings, strict=True)):
        validate(calibration, run["device_fingerprint"], digest=evaluator_digest())
        if run["result_sha256"] != binding["result_sha256"]:
            raise ValueError("Actual paired activity belongs to a different measurement result")
        rows = [row for row in paired["pairs"] if row.get("repeat", 0) == repeat]
        summary = paired_summary(rows)
        totals = {role: {key: sum(row[key] for row in run["calls"] if row["role"] == role)
                         for key in ("wall_ms", "device_busy_ms", "kernel_count", "activity_count")} for role in "RCB"}
        item = launch_only(totals["R"], totals["C"], launch_ms=calibration["launch_median_ms"],
                           speedup_ci95=summary["speedup_ci95"], noise_ci95=summary["reference_control_ci95"])
        groups.append({**item, "repeat": repeat, "roles": totals, "origin_receipt_sha256": run["origin_receipt_sha256"],
                       "result_sha256": run["result_sha256"]})
    flags = [row["launch_only"] for row in groups]
    return {"protocol": "kernelbench_pro_scored_launch_attribution_v1", "launch_only": all(flags),
            "mixed_repeat_classification": any(flags) and not all(flags), "repeats": groups,
            "calibration_sha256": calibration["receipt_sha256"], "paired_sha256": fingerprint(paired),
            "source": "actual_native_paired_calls", "scored_wall_times_replaced": False,
            "classification_is_diagnostic": False, "algorithm_or_efficiency_gain_proven": False}

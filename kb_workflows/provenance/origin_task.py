"""Native call coverage and separate control/profile origin diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import weakref
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from ..common import atomic_json, evaluator_digest, failure, fingerprint, read_json
from .kernel_origin import MAX_RECORDS, _sha, summarize
from .submission_source import SourceRegistry, record_sources

NATIVE_PROTOCOL = "kernelbench_pro_native_origin_v2"
NATIVE_SCOPE = "native_correctness_both_orders_warmups_timing_memory_and_paired; model_state_bound; independent_diagnostics_excluded"
_ACTIVE = ContextVar("kernelbench_native_origin_calls", default=None)
MAX_REPORT_BYTES = 64 * 1024**2


def _validate(payload):
    settings = payload["eval_config"]
    target = payload.get("_origin_target", payload.get("task", "evaluate"))
    role = payload.get("_origin_role", "candidate")
    if role not in {"candidate", "reference"} or role == "reference" and target != "paired_timing":
        raise ValueError("Reference origin/counters require the explicit native paired baseline")
    if settings.get("device_backend", "cuda") != "cuda":
        raise ValueError("Native origin capture supports CUDA; Neuron association is pending")
    if target not in {"evaluate", "paired_timing"} or settings.get("use_hidden_tests") and target != "evaluate":
        raise ValueError("Origin capture requires native evaluation or public paired timing")
    correct = settings.get("num_correct_trials", 0)
    if target == "evaluate" and (type(correct) is not int or not 1 <= correct <= MAX_RECORDS // 2):
        raise ValueError("Origin coverage requires a bounded positive native correctness count")
    pairs = settings.get("paired_trials", 30)
    if target == "paired_timing" and (type(pairs) is not int or not 10 <= pairs <= MAX_RECORDS - 3):
        raise ValueError("Origin coverage requires a bounded native paired trial count")
    if "origin_seeds" in settings:
        raise ValueError("Three-seed origin loops are retired; use the complete native workflow")
    performance = target == "evaluate" and settings.get("measure_performance", not settings.get("use_hidden_tests"))
    trials = settings.get("num_perf_trials", 0)
    if performance and (type(trials) is not int or trials <= 0):
        raise ValueError("Native origin requires a positive performance trial count")
    count = 3 + trials if performance else 0
    seeds = payload.get("_origin_timing_seeds")
    if (not isinstance(seeds, list) or not 0 <= count <= MAX_RECORDS or len(seeds) != count
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Origin capture requires the complete frozen native timing seed plan")
    threshold = settings.get("origin_minimum_custom_fraction", 0.95)
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Origin diagnostic fraction must be in (0, 1]")
    return target, seeds, threshold


def native_request(payload):
    """Normalize instrumentation and transport without changing native settings."""
    ignored = {"_origin_target", "_origin_stage", "_native_counter", "_scored_result", "request_id", "build_root"}
    result = copy.deepcopy({key: value for key, value in payload.items() if key not in ignored})
    result["task"] = payload.get("_origin_target", payload.get("task", "evaluate"))
    return result


class _Calls:
    def __init__(self, capture, seeds, role="candidate", *, scored=False, registry=None):
        self.capture, self.seeds = capture, seeds
        self.registry = registry or SourceRegistry()
        self.scored = scored
        self.role = role
        self.rows, self.windows, self.changes = [], {}, []
        self.timings = []
        self.cursor, self.last_seed = 0, None

    def model_state(self, model, args):
        from ..evaluation.state_paths import invocation_state

        try:
            return invocation_state(model, args, callable_identity=self.registry.callable_identity)
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as exc:
            return {"status": "not_measured", "reason": str(exc)[:300]}

    def next_seed(self):
        if self.scored:
            from ..evaluation.measurement import _SEED

            if len(self.seeds) >= MAX_RECORDS:
                raise ValueError("Excessive native timing seed coverage")
            self.seeds.append(_SEED(63))
        if self.cursor >= len(self.seeds):
            raise ValueError("Native timing requested more inputs than its frozen plan")
        self.last_seed = self.seeds[self.cursor]
        self.cursor += 1
        return self.last_seed

    def call(self, model, args, *, device, phase, seed=None, label=None, order=None):
        import torch

        from src.input_groups import layout

        from ..evaluation.determinism import signature
        from ..evaluation.measurement import input_signature, synchronize
        from ..evaluation.strict_validation import environment_guard

        seed = self.last_seed if seed is None else seed
        if type(seed) is not int or seed < 0 or len(self.rows) >= MAX_RECORDS:
            raise ValueError("Missing native seed or excessive native call coverage")
        geometry = input_signature(args)
        aliases = layout(args)
        if aliases is None:
            raise ValueError("Native origin cannot describe input storage, offsets and aliases")
        row = {"id": len(self.rows) + 1, "role": self.role, "phase": phase, "seed": seed, "label": label, "order": order,
               "input_content_sha256": fingerprint({"value": signature(args, typed=True)}),
               "input_signature_sha256": fingerprint({"signature": geometry}),
               "input_layout_sha256": fingerprint(aliases),
               "called": False, "verified": False, "output_content_sha256": None,
               "model_state_before": self.model_state(model, args)}
        self.rows.append(row)
        if self.scored:
            # This is the callable used by the actual timer. Hashing and mode
            # observations surround its scope; no forwarding wrapper or extra
            # synchronization is inserted into the measured interval.
            row.update(called=True, training=model.training, no_grad=not torch.is_grad_enabled())
            return model, row

        def execute(*values):
            if row["called"]:
                raise ValueError("A native call scope cannot execute a second forward")
            row["called"] = True
            training = model.training
            if type(training) is not bool:
                raise ValueError("Candidate training mode must be an explicit boolean")
            grad = torch.is_grad_enabled()
            scope = self.capture.call(seed=seed, signature=geometry, training=training) if self.capture else nullcontext(None)
            with environment_guard(self.changes), scope as window:
                output = model(*values)
                synchronize(device)
                row.update(training=training, training_after=model.training,
                           no_grad=not grad and not torch.is_grad_enabled())
                if window is not None:
                    window.update(training_after=row["training_after"], no_grad=row["no_grad"])
                    self.windows[row["id"]] = window
            return output
        return execute, row


def next_seed():
    state = _ACTIVE.get()
    return state.next_seed() if state is not None else None


@contextmanager
def candidate_call(model, args, *, role="candidate", **scope):
    """Inactive calls retain the exact original callable, with no timed wrapper."""
    state = _ACTIVE.get()
    if state is None or state.role != role:
        yield model, None
        return
    execute, row = state.call(model, args, **scope)
    try:
        yield execute, row
    finally:
        row["model_state_after"] = state.model_state(model, args)
        if state.scored:
            import torch

            row.update(training_after=model.training, no_grad=row["no_grad"] and not torch.is_grad_enabled())


def snapshot_output(row, output):
    if row is not None:
        from ..evaluation.determinism import signature

        row["output_content_sha256"] = fingerprint({"value": signature(output, typed=True)})


def record_elapsed(row, milliseconds):
    """Keep the native timer outside cross-process observable comparisons."""
    if row is not None:
        _ACTIVE.get().timings.append((row, milliseconds))


def confirm(row):
    if row is not None:
        if row["output_content_sha256"] is None:
            raise ValueError("Native output was not snapshotted before verification")
        row["verified"] = True
        window = _ACTIVE.get().windows.get(row["id"])
        if window is not None:
            window["output_verified"] = True
            window["native_call_sha256"] = fingerprint(row)


def native_policy(function):
    @wraps(function)
    def wrapped(payload, *args, **kwargs):
        stage = payload.get("_origin_stage")
        if stage is None:
            settings = payload["eval_config"]
            if (settings.get("evidence_level") in {"L1", "L2"} and not settings.get("diagnostic_stage")
                    and settings.get("device_backend", "cuda") == "cuda"
                    and payload.get("task", "evaluate") in {"evaluate", "paired_timing"}):
                return _run_scored(payload, lambda: function(payload, *args, **kwargs))
            return function(payload, *args, **kwargs)
        if stage not in {"control", "profile", "counter"}:
            raise ValueError("Unknown native origin worker stage")
        observer = None
        if stage == "counter":
            from .counter_task import Profiler

            observer = Profiler(payload)
        return _run_native(payload, lambda: function(payload, *args, **kwargs), observer=observer)
    return wrapped


def _document(payload, result, state, stage):
    binding = {"native_request_sha256": fingerprint(native_request(payload)), "role": state.role,
               "evaluator_sha256": payload["eval_config"].get("evaluator_sha256"),
               "device_fingerprint": result.get("metadata", {}).get("device_fingerprint"), "scope": NATIVE_SCOPE}
    return {"protocol": NATIVE_PROTOCOL, "stage": stage, "binding": binding,
            "native_result": result, "calls": state.rows, "seeds_consumed": state.cursor,
            "native_timings": [{"id": row["id"], "native_call_sha256": fingerprint(row), "wall_ms": elapsed}
                               for row, elapsed in state.timings],
            "environment_changes": sorted(set(state.changes))}


def _run_scored(payload, invoke):
    if _ACTIVE.get() is not None or "_origin_timing_seeds" in payload:
        raise ValueError("Scored calls cannot inherit a diagnostic observer or predeclared timing seed plan")
    request = native_request(payload)
    state = _Calls(None, [], scored=True)
    token = _ACTIVE.set(state)
    try:
        with record_sources(state.registry):
            result = invoke()
    finally:
        _ACTIVE.reset(token)
    frozen = {**request, "_origin_timing_seeds": state.seeds}
    if not state.registry.intact():
        state.changes.append("registered_source")
    document = _document(frozen, result, state, "scored")
    document.pop("native_result")
    document.update(native_request=request, timing_seed_plan=state.seeds,
                    timed_callable_wrapped=False, execution_origin_collected=False)
    document["receipt_sha256"] = fingerprint(document)
    result.setdefault("metadata", {})["native_call_receipt"] = document
    return result


def _run_native(payload, invoke, *, observer=None):
    _, seeds, _ = _validate(payload)
    from src import eval as native

    from ..evaluation.strict_validation import environment_guard
    from .cuda_origin import Capture
    from .origin_binaries import Catalog

    role = payload.get("_origin_role", "candidate")
    source = payload["ref_arch_src"] if role == "reference" else payload["kernel_src"]
    capture = Capture(source, candidate_windows_only=True) if payload["_origin_stage"] == "profile" else None
    catalog = Catalog() if capture else None
    state = _Calls(observer or capture, seeds, role, registry=capture.registry if capture else None)
    load, move = native.load_custom_model, native.move_model
    candidates = {}

    def guarded_load(*args, **kwargs):
        if capture is not None and capture.extensions is not None:
            capture.extensions.prepare()  # Native entrypoint has now set its build environment.
        with environment_guard(state.changes):
            cls = load(*args, **kwargs)
        def construct(*values, **options):
            with environment_guard(state.changes):
                model = cls(*values, **options)
            candidates[id(model)] = weakref.ref(model)
            return model
        return construct

    def guarded_move(model, *args, **kwargs):
        owner = candidates.get(id(model))
        if owner is None or owner() is not model:
            return move(model, *args, **kwargs)
        with environment_guard(state.changes):
            result = move(model, *args, **kwargs)
        candidates[id(result)] = weakref.ref(result)
        return result

    native.load_custom_model, native.move_model = guarded_load, guarded_move
    token = _ACTIVE.set(state)
    try:
        with tempfile.TemporaryDirectory(prefix="kb-origin-control-cache-") as cache:
            previous = os.environ.get("TRITON_CACHE_DIR")
            try:
                if not capture:
                    os.environ["TRITON_CACHE_DIR"] = cache
                try:
                    with capture.recording() if capture else record_sources(state.registry):
                        result = invoke()
                except Exception as exc:
                    result = failure(str(exc), error_name=type(exc).__name__)
            finally:
                if previous is None:
                    os.environ.pop("TRITON_CACHE_DIR", None)
                else:
                    os.environ["TRITON_CACHE_DIR"] = previous
    finally:
        if native.load_custom_model is not guarded_load or native.move_model is not guarded_move:
            state.changes.append("native_loader_or_move_hook")
        native.load_custom_model, native.move_model = load, move
        _ACTIVE.reset(token)
    if evaluator_digest() != payload["eval_config"].get("evaluator_sha256"):
        state.changes.append("evaluator_source")
    if not state.registry.intact():
        state.changes.append("registered_source")
    document = _document(payload, result, state, payload["_origin_stage"])
    binding = document["binding"]
    if observer is not None:
        document["counter_windows"] = observer.windows
        document["pid"] = observer.pid
        document["counter_simulated"] = observer.simulated
    if capture:
        if capture.extensions is not None:
            capture.bind_extensions(capture.extensions.collect(timeout=min(30, payload["eval_config"]["timeout"] / 8)))
        capture.raw["binary_inventory"] = catalog.collect(timeout=min(60, payload["eval_config"]["timeout"] / 4))
        capture.raw["binding"] = binding
        capture.raw["environment_changes"] = document["environment_changes"]
        if state.changes:
            capture.error("Native candidate changed process settings or installed hooks")
        document["raw"] = capture.raw
    return document


def _coverage(document, payload):
    """Replay actual native receipts, rather than trusting a claimed call count."""
    target, seeds, _ = _validate(payload)
    result = document["native_result"]
    if result.get("compiled") is not True or result.get("correctness") is not True:
        raise ValueError("Native origin run failed compilation or correctness")
    metadata = result["metadata"]
    if target == "evaluate":
        from src.eval import correctness_seeds
        from src.reliability import HIDDEN_SEEDS

        checks = metadata["reliability_checks"]
        count = payload["eval_config"]["num_correct_trials"]
        expected_standard = list(enumerate(correctness_seeds(payload["eval_config"].get("evaluation_seed", 42), count), 1))
        if [(check["label"], check["seed"]) for check in checks[:count]] != [(f"standard_{index}", seed) for index, seed in expected_standard]:
            raise ValueError("Missing or rebound native standard correctness cases")
        if payload["eval_config"].get("use_hidden_tests"):
            hidden_seeds = payload["eval_config"].get("hidden_seeds", list(HIDDEN_SEEDS))
            hidden = checks[count:]
            if (metadata.get("hidden_validation_seeds") != hidden_seeds or metadata.get("hidden_failed_configs") != []
                    or metadata.get("hidden_completed_checks") != len(hidden) or not hidden
                    or {check["seed"] for check in hidden} != set(hidden_seeds)):
                raise ValueError("Missing or changed native hidden case/seed coverage")
            if payload["eval_config"].get("operator_inputs"):
                from ..numerics.operator_input_policy import issue

                if reason := issue(result, payload):
                    raise ValueError(reason)
        elif len(checks) != count:
            raise ValueError("Public origin capture cannot import extra hidden cases")
        expected = [("correctness", check["seed"], check["label"], order)
                    for check in checks for order in check["orders"]]
        if (not expected or any(not check["passed"] or check["orders"] != ["candidate_first", "reference_first"]
                                for check in metadata["reliability_checks"])):
            raise ValueError("Native origin lacks complete paired correctness receipts")
        expected += [("warmup" if index < 3 else "timing", seed, str(index), None) for index, seed in enumerate(seeds)]
        if seeds and metadata.get("timing_receipt", {}).get("input_seeds") != seeds:
            raise ValueError("Native timing did not use its complete frozen input plan")
        if seeds:
            expected.append(("memory", payload["eval_config"].get("evaluation_seed", 42), "memory_probe", None))
    else:
        from ..evaluation.paired_task import call_plan

        expected = [("paired_warmup" if index < 3 else "paired", seed, str(index), order)
                    for index, seed, order in call_plan(trials=payload["eval_config"].get("paired_trials", 30),
                                                       warmup=3, seed=payload["eval_config"].get("evaluation_seed", 42))]
        paired = metadata["paired_timing"]
        if (paired.get("num_pairs") != len(expected) - 3 or [(row["seed"], row["order"]) for row in paired["pairs"]]
                != [(seed, order) for _, seed, _, order in expected[3:]]):
            raise ValueError("Paired native results do not match the captured seed/order schedule")
    rows = document["calls"]
    if ([(row["phase"], row["seed"], row["label"], row["order"]) for row in rows] != expected
            or [row["id"] for row in rows] != list(range(1, len(expected) + 1))
            or document["seeds_consumed"] != len(seeds) or document["environment_changes"]
            or any(row.get("role") != payload.get("_origin_role", "candidate")
                   or row["called"] is not True or row["verified"] is not True or row.get("no_grad") is not True
                   or type(row.get("training")) is not bool or row.get("training_after") != row["training"]
                   or any(not _sha(row.get(key)) for key in ("input_content_sha256", "input_signature_sha256",
                                                             "input_layout_sha256", "output_content_sha256"))
                   or any(not isinstance(row.get(key), dict) or row[key].get("status") != "observed"
                          or not _sha(row[key].get("sha256")) or type(row[key].get("tensor_count")) is not int
                          or row[key]["tensor_count"] < 0
                          or row[key].get("scope") != "module_data_entrypoints_and_input_state_aliases; no_host_control_proof"
                          for key in ("model_state_before", "model_state_after")) for row in rows)):
        raise ValueError("Missing, changed, unverified or mode-unbound native candidate calls")
    timed = [row for row in rows if row["phase"] in {"warmup", "timing", "paired_warmup", "paired"}]
    timings = document["native_timings"]
    if (not isinstance(timings, list) or len(timings) != len(timed)
            or any(item.get("id") != row["id"] or item.get("native_call_sha256") != fingerprint(row)
                   or type(item.get("wall_ms")) not in (int, float) or not math.isfinite(item["wall_ms"]) or item["wall_ms"] <= 0
                   for item, row in zip(timings, timed))):
        raise ValueError("Native timer omitted or rebound a measured/warmup call")
    measured = [item["wall_ms"] for item, row in zip(timings, timed) if row["phase"] in {"timing", "paired"}]
    if target == "evaluate" and seeds:
        receipt = metadata["timing_receipt"]
        if (result.get("runtime_stats", {}).get("samples_ms") != measured
                or receipt.get("first_timed_call_ms") != timings[0]["wall_ms"]
                or not math.isclose(receipt.get("warmup_total_ms", -1), sum(item["wall_ms"] for item in timings[:3]), rel_tol=1e-12)):
            raise ValueError("Native performance samples differ from their observed candidate calls")
        memory, row = metadata["memory_receipt"], rows[-1]
        peak = memory.get("peak_bytes")
        raw_peak, copy_bytes = memory.get("raw_peak_bytes"), memory.get("verification_input_bytes")
        if (type(peak) is not int or peak < 0 or memory.get("input_seed") != row["seed"]
                or type(raw_peak) is not int or type(copy_bytes) is not int or copy_bytes < 0 or raw_peak != peak + copy_bytes
                or memory.get("output_verified") is not True or memory.get("verification_outside_peak") is not True
                or fingerprint({"signature": memory.get("input_signature")}) != row["input_signature_sha256"]
                or result.get("peak_memory") != peak
                or any(result.get("memory_stats", {}).get(key) != memory[key]
                       for key in ("peak_bytes", "raw_peak_bytes", "verification_input_bytes"))):
            raise ValueError("Native memory probe omitted verification, inputs or peak measurement binding")
    key = "reference_ms" if payload.get("_origin_role") == "reference" else "candidate_ms"
    if target == "paired_timing" and [pair[key] for pair in metadata["paired_timing"]["pairs"]] != measured:
        raise ValueError("Native paired samples differ from their observed candidate calls")
    return rows


def validate_native(document, payload, *, stage, simulated=False):
    """Replay one native result, including counter-stage full call sequences."""
    if payload["eval_config"].get("evaluator_sha256") != evaluator_digest():
        raise ValueError("Native origin stages or evaluator version differ")
    binding = document["binding"]
    metadata = document["native_result"]["metadata"]
    if (document.get("protocol") != NATIVE_PROTOCOL or document.get("stage") != stage
                or binding.get("native_request_sha256") != fingerprint(native_request(payload))
                or binding.get("role") != payload.get("_origin_role", "candidate")
                or binding.get("evaluator_sha256") != evaluator_digest()
                or binding.get("scope") != NATIVE_SCOPE
                or not isinstance(binding.get("device_fingerprint"), dict)
                or binding["device_fingerprint"] != metadata.get("device_fingerprint")
                or metadata.get("reference_sha256") != fingerprint({"source": payload["ref_arch_src"]})
                or metadata.get("kernel_sha256") != fingerprint({"source": payload["kernel_src"]})
                or metadata.get("precision") != payload["eval_config"]["precision"]):
        raise ValueError("Native origin stage changed its source, device or request binding")
    if not simulated:
        from ..analysis.calibration import identity

        identity(binding["device_fingerprint"])
    return _coverage(document, payload)


def _scored_contract(payload):
    request = native_request(payload)
    request.pop("_origin_timing_seeds", None)
    request["eval_config"].pop("origin_minimum_custom_fraction", None)
    return request


def validate_scored(result, payload=None, *, allow_simulated=False):
    """Replay actual timer observations without granting execution ownership."""
    from ..evaluation.evaluate_task import validate_sources

    try:
        record = result["metadata"]["native_call_receipt"]
        request = record["native_request"]
        if (record.get("timed_callable_wrapped") is not False or record.get("execution_origin_collected") is not False
                or record.get("receipt_sha256") != fingerprint({k: v for k, v in record.items() if k != "receipt_sha256"})
                or request.get("task", "evaluate") not in {"evaluate", "paired_timing"}
                or "_origin_timing_seeds" in request or request.get("_origin_role", "candidate") != "candidate"
                or request["eval_config"].get("evidence_level") not in {"L1", "L2"}
                or request["eval_config"].get("diagnostic_stage")
                or payload is not None and _scored_contract(request) != _scored_contract(payload)):
            raise ValueError("Scored native calls changed their receipt, request or observation contract")
        validate_sources(request)
        frozen = {**request, "_origin_timing_seeds": record["timing_seed_plan"]}
        document = {**record, "native_result": result}
        validate_native(document, frozen, stage="scored", simulated=allow_simulated)
        return frozen, document
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Missing or invalid actual scored native calls") from exc


def read_scored(path, problem, sample, task="evaluate"):
    """Read one existing native artifact for both origin and counter previews."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Scored results must be an existing regular JSON file")
    with path.open("rb") as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("Scored results exceed the report byte limit")
    result = json.loads(data)
    if isinstance(result, dict) and "metadata" not in result:
        result = result[f"{problem}_{sample}" + (":0" if task == "paired_timing" else "")]
    return result


def _replay_parts(payload, stages, raw, *, simulated):
    """One verifier for collection and saved-report analysis; no cached verdicts."""
    _, _, threshold = _validate(payload)
    if type(simulated) is not bool:
        raise ValueError("Native origin simulation state must be explicit")
    simulated = simulated or raw.get("simulated") is not False
    if set(stages) != {"control", "profile"}:
        raise ValueError("Native origin control/profile stages are incomplete")
    control, profiled = (validate_native(stages[stage], payload, stage=stage, simulated=simulated) for stage in ("control", "profile"))
    if control != profiled or stages["control"]["binding"] != stages["profile"]["binding"]:
        raise ValueError("Control/profile native input, output, mode observations or device differ")
    source = payload["ref_arch_src"] if payload.get("_origin_role") == "reference" else payload["kernel_src"]
    if (raw.get("candidate_sha256") != hashlib.sha256(source.encode()).hexdigest()
            or raw.get("candidate_windows_only") is not True or raw.get("binding") != stages["profile"]["binding"]
            or len(raw.get("windows", [])) != len(profiled)
            or any(window.get("id") != row["id"] or window.get("native_call_sha256") != fingerprint(row)
                   or window.get("seed") != row["seed"] or window.get("output_verified") is not True
                   or fingerprint({"signature": window.get("input_signature")}) != row["input_signature_sha256"]
                   or any(window.get(key) != row[key] for key in ("training", "training_after", "no_grad"))
                   for window, row in zip(raw["windows"], profiled))):
        raise ValueError("CUPTI windows do not bind every validated native candidate call")
    coverage = {"status": "passed", "candidate_calls": len(control), "bitwise_observable_replay": True,
                "simulated": simulated,
                "memory_probe_included": bool(payload.get("_origin_target", payload.get("task")) == "evaluate"
                                               and payload.get("_origin_timing_seeds")),
                "uninstrumented_path_equivalence_proven": False}
    if "_scored_result" in payload:
        _, scored = validate_scored(payload["_scored_result"], payload, allow_simulated=simulated)
        if (scored["calls"] != control
                or scored["binding"]["device_fingerprint"] != stages["control"]["binding"]["device_fingerprint"]):
            raise ValueError("Origin replay differs from actual scored calls, state, entrypoints, outputs, modes or device")
        coverage["scored_call_observations_match"] = True
    return coverage, summarize(raw, minimum_custom_fraction=threshold)


def replay(report, *, allow_simulated=False):
    """Recompute a saved native report without starting a collector or workload."""
    from ..evaluation.evaluate_task import validate_sources

    try:
        if (not isinstance(report, dict) or report.get("protocol") != NATIVE_PROTOCOL or report.get("status") != "ok"
                or report.get("purpose") != "native_execution_origin_diagnostics_only"
                or report.get("formal_qualification_implemented") is not False or report.get("scored_wall_times_replaced") is not False
                or report.get("receipt_sha256") != fingerprint({k: v for k, v in report.items() if k not in {"receipt_sha256", "metadata"}})):
            raise ValueError("Native origin report is missing, changed or uses an unsupported scope")
        payload = report["native_request"]
        validate_sources(payload)
        coverage, summary = _replay_parts(payload, report["native_runs"], report["raw"], simulated=report["native_coverage"]["simulated"])
        if (coverage != report["native_coverage"] or summary != report["summary"]
                or coverage["simulated"] and not allow_simulated):
            raise ValueError("Saved origin coverage or summary is stale, changed or simulated")
        return summary
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Incomplete or invalid native origin report") from exc


def audit(payload, *, run=None, budget_seconds=None):
    from ..evaluation.evaluate_task import validate_sources
    from ..evaluation.tool_runner import environment, process

    target, _, _ = _validate(payload)
    validate_sources(payload)
    if "_scored_result" in payload:
        validate_scored(payload["_scored_result"], payload, allow_simulated=run is not None)
    if run is None:
        from importlib.metadata import version

        from .cuda_origin import abi_supported

        if (not abi_supported() or any(version(package).split("+")[0] != expected for package, expected in (
                ("cupti-python", "13.3.0"), ("triton", "3.7.0"), ("torch", "2.12.0")))):
            raise ValueError("Native origin requires its pinned CUPTI/Triton/Torch packages and Linux ABI")
    budget = payload["eval_config"]["timeout"] if budget_seconds is None else budget_seconds
    if not math.isfinite(budget) or budget > payload["eval_config"]["timeout"]:
        raise ValueError("Origin collection cannot exceed its request budget")
    limit = (budget - 10) / 2
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Origin control/profile need positive budgets including process cleanup")
    runner, stages = run or process, {}
    deadline = time.monotonic() + budget

    def failed(reason, directory):
        log = directory / "worker.log"
        prefix = ""
        if log.is_file() and not log.is_symlink():
            with log.open("rb") as stream:
                prefix = stream.read(4096).decode(errors="replace")
        return {"status": "not_measured", "reason": reason, "completed_stages": stages,
                "worker_log_prefix": prefix, "scored_wall_times_replaced": False, "formal_qualification_implemented": False}

    with tempfile.TemporaryDirectory(prefix="kb-native-origin-") as temporary:
        root = Path(temporary)
        for stage in ("control", "profile"):
            directory = root / stage
            directory.mkdir()
            child = copy.deepcopy(payload)
            child.update(task=target, _origin_stage=stage, build_root=str(directory / "build"))
            child.pop("_origin_target", None)
            child.pop("_scored_result", None)
            child["request_id"] = "kb-" + fingerprint(child)
            source, output = directory / "request.json", directory / "result.json"
            atomic_json(source, child)
            outcome = runner([sys.executable, "-m", "kb_workflows.evaluation.evaluate_task", str(source), str(output), "--native-worker"],
                             min(limit, max(0, deadline - time.monotonic())), directory / "worker.log", environment=environment())
            if (outcome.get("exit_code") != 0 or outcome.get("timed_out") is not False or not output.is_file()
                    or output.is_symlink() or output.stat().st_size > MAX_REPORT_BYTES):
                return failed(f"Native origin {stage} worker failed or report is missing/excessive", directory)
            try:
                document = read_json(output)
            except (ValueError, OSError) as exc:
                return failed(f"Native origin {stage} result cannot be decoded: {exc}", directory)
            if (not isinstance(document, dict) or document.get("protocol") != NATIVE_PROTOCOL or document.get("stage") != stage
                    or not isinstance(document.get("binding"), dict)
                    or document.get("binding", {}).get("native_request_sha256") != fingerprint(native_request(child))):
                stages[stage] = document
                return failed("Native origin worker changed its request binding or stage", directory)
            document["worker_wall_seconds"] = outcome.get("request_wall_seconds")
            stages[stage] = document
        if not isinstance(stages["profile"].get("raw"), dict):
            return failed("Profile worker omitted its raw source/launch capture", root / "profile")
    raw = stages["profile"].pop("raw")
    try:
        coverage, summary = _replay_parts(payload, stages, raw, simulated=run is not None)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        coverage = {"status": "not_measured", "reason": str(exc)}
        summary = {"status": "not_measured", "reason": str(exc), "timings_are_diagnostic": True,
                   "formal_qualification_implemented": False}
    request = copy.deepcopy(payload)
    request.pop("request_id", None)
    request.pop("build_root", None)
    report = {"protocol": NATIVE_PROTOCOL, "status": "ok", "purpose": "native_execution_origin_diagnostics_only", "raw": raw,
              "native_request": request, "native_runs": stages, "native_coverage": coverage, "summary": summary,
              "scored_wall_times_replaced": False, "formal_qualification_implemented": False}
    report["receipt_sha256"] = fingerprint(report)
    return report

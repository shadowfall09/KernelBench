"""Actual scored source evidence, native call coverage and separate replay diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import weakref
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from ..common import atomic_json, evaluator_digest, failure, fingerprint, read_json
from .kernel_origin import MAX_RECORDS, _sha, summarize
from .kernel_origin import PROTOCOL as RAW_PROTOCOL
from .submission_source import SourceRegistry, record_sources

NATIVE_PROTOCOL = "kernelbench_pro_native_origin_v2"
NATIVE_SCOPE = "native_correctness_both_orders_warmups_timing_memory_and_paired; model_state_bound; independent_diagnostics_excluded"
_ACTIVE = ContextVar("kernelbench_native_origin_calls", default=None)
MAX_REPORT_BYTES = 64 * 1024**2
SCORED_CAPTURE = "kernelbench_pro_scored_capture_v1"
SCORED_MODE = "synchronized_wall_continuous_cupti_v1"
BASELINE_SCOPE = "baseline_all_device_activities; no_candidate_origin_claim"
PAIRED_CAPTURE = "kernelbench_pro_paired_device_windows_v1"


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
        self.timings, self.paired_windows = [], []
        self.open_windows = {}
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
            if self.capture is not None:
                scope = self.capture.call(seed=seed, signature=geometry, training=row["training"])
                self.windows[row["id"]] = scope.__enter__()
                self.open_windows[row["id"]] = scope
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
def native_calls(registry, capture=None):
    """Share original-call/input/state observations with native device backends."""
    if _ACTIVE.get() is not None:
        raise ValueError("Native invocation observations cannot be nested")
    state = _Calls(capture, [], scored=True, registry=registry)
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


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
        if state.scored:
            import torch

            window = state.windows.get(row["id"])
            if window is not None:
                state.open_windows.pop(row["id"]).__exit__(*sys.exc_info())
            row.update(training_after=model.training, no_grad=row["no_grad"] and not torch.is_grad_enabled())
            if window is not None:
                window.update(training_after=row["training_after"], no_grad=row["no_grad"])
        row["model_state_after"] = state.model_state(model, args)


@contextmanager
def paired_device_call(role, trial, seed, order, args):
    """Observe all three actual paired calls; timestamps surround the native timer."""
    state = _ACTIVE.get()
    if state is None or not state.scored or state.capture is None:
        yield None
        return
    from ..evaluation.measurement import input_signature

    if len(state.paired_windows) >= MAX_RECORDS:
        raise ValueError("Excessive paired device windows")
    row = {"id": len(state.paired_windows) + 1, "role": role, "trial": trial, "seed": seed, "order": order,
           "device": 0, "input_signature_sha256": fingerprint({"signature": input_signature(args)}),
           "output_verified": False, "start_ns": int(state.capture.timestamp()), "end_ns": None}
    state.paired_windows.append(row)
    try:
        yield row
    finally:
        row["end_ns"] = int(state.capture.timestamp())


def confirm_paired_device(row, elapsed, output, observed):
    if row is not None:
        from ..evaluation.determinism import signature

        row.update(wall_ms=elapsed, output_verified=True,
                   output_content_sha256=observed["output_content_sha256"] if observed is not None else
                       fingerprint({"value": signature(output, typed=True)}))


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
                    and payload.get("task", "evaluate") in {"evaluate", "baseline", "paired_timing"}):
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
               "device_fingerprint": result.get("metadata", {}).get("device_fingerprint", result.get("device_fingerprint")),
               "scope": BASELINE_SCOPE if payload.get("task") == "baseline" else NATIVE_SCOPE}
    return {"protocol": NATIVE_PROTOCOL, "stage": stage, "binding": binding, "pid": os.getpid(),
            "native_result": result, "calls": state.rows, "seeds_consumed": state.cursor,
            "native_timings": [{"id": row["id"], "native_call_sha256": fingerprint(row), "wall_ms": elapsed}
                               for row, elapsed in state.timings],
            "environment_changes": sorted(set(state.changes))}


@contextmanager
def _source_guards(state, capture):
    from src import eval as native

    from ..evaluation.strict_validation import environment_guard

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
    try:
        yield
    finally:
        if native.load_custom_model is not guarded_load or native.move_model is not guarded_move:
            state.changes.append("native_loader_or_move_hook")
        native.load_custom_model, native.move_model = load, move


def _run_scored(payload, invoke):
    from ..evaluation.tool_runner import exclusive_compute

    if _ACTIVE.get() is not None or "_origin_timing_seeds" in payload:
        raise ValueError("Scored calls cannot inherit a diagnostic observer or predeclared timing seed plan")
    request = native_request(payload)
    from .cuda_origin import Capture
    from .origin_binaries import Catalog

    capture, catalog, unavailable = None, None, None
    with ExitStack() as stack:
        try:
            capture = Capture(payload.get("kernel_src", payload["ref_arch_src"]), candidate_windows_only=False)
            catalog = Catalog()
            stack.enter_context(capture.recording())
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            # A missing optional collector cannot turn a numerical answer into
            # a failure. No candidate has run yet; do not retry invoke().
            unavailable = f"{type(exc).__name__}: {exc}"[:500]
            capture = catalog = None
        state = _Calls(capture, [], scored=True, registry=capture.registry if capture else None)
        token = _ACTIVE.set(state)
        try:
            with _source_guards(state, capture), record_sources(state.registry), exclusive_compute(payload["eval_config"]) as exclusive:
                result = invoke()
        finally:
            _ACTIVE.reset(token)
    frozen = {**request, "_origin_timing_seeds": state.seeds}
    if not state.registry.intact():
        state.changes.append("registered_source")
    document = _document(frozen, result, state, "scored")
    document["exclusive_compute"] = exclusive
    if capture is not None:
        _finish_capture(capture, catalog, document, payload)
    origin = {"protocol": SCORED_CAPTURE, "mode": SCORED_MODE,
              "status": "collected" if capture is not None else "not_measured", "error": unavailable,
              "binding": document["binding"], "timing_sha256": fingerprint(_scored_timing(result)),
              "exclusive_compute_sha256": fingerprint(exclusive) if exclusive is not None else None,
              "raw": capture.raw if capture is not None else None}
    if payload.get("task") == "paired_timing":
        origin["paired_capture"] = {"protocol": PAIRED_CAPTURE, "calls": state.paired_windows}
    if payload.get("task") == "baseline":
        origin.update(native_request=request, timing_seed_plan=state.seeds, exclusive_compute=exclusive)
    origin["receipt_sha256"] = fingerprint(origin)
    result.setdefault("metadata", {})["origin_capture"] = origin
    if payload.get("task") == "baseline":
        return result
    document.pop("native_result")
    document.update(native_request=request, timing_seed_plan=state.seeds,
                    timed_callable_wrapped=False, execution_origin_collected=capture is not None,
                    origin_capture_sha256=origin["receipt_sha256"])
    document["receipt_sha256"] = fingerprint(document)
    result.setdefault("metadata", {})["native_call_receipt"] = document
    from ..evaluation.evidence import from_result

    metadata = result["metadata"]
    additional = {**metadata.get("evidence_receipt", {}).get("checks", {}),
                  **metadata.get("paired_checks", {}), "kernel_origin": origin_check(result)}
    metadata["evidence_receipt"] = from_result(result, payload["eval_config"],
        scope="hidden" if payload["eval_config"].get("use_hidden_tests") else "standard", additional=additional)
    return result


def _scored_timing(result):
    metadata = result.get("metadata", {})
    return {"runtime": result.get("runtime", result.get("mean")),
            "samples_ms": result.get("runtime_stats", result).get("samples_ms"),
            "timing_receipt": metadata.get("timing_receipt", result.get("timing_receipt")),
            "paired_timing": metadata.get("paired_timing"),
            "cold_start_ms": metadata.get("cold_start_ms", result.get("cold_start_ms"))}


def _finish_capture(capture, catalog, document, payload):
    try:
        if capture.extensions is not None:
            capture.bind_extensions(capture.extensions.collect(timeout=min(30, payload["eval_config"]["timeout"] / 8)))
        capture.raw["binary_inventory"] = catalog.collect(timeout=min(60, payload["eval_config"]["timeout"] / 4))
    except (OSError, RuntimeError, ValueError) as exc:
        capture.error(f"Binary inventory: {type(exc).__name__}: {exc}"[:500])
    capture.raw["binding"] = document["binding"]
    capture.raw["environment_changes"] = document["environment_changes"]
    if document["environment_changes"]:
        capture.error("Native candidate changed process settings or installed hooks")


def _run_native(payload, invoke, *, observer=None):
    from ..evaluation.tool_runner import exclusive_compute

    _, seeds, _ = _validate(payload)
    from .cuda_origin import Capture
    from .origin_binaries import Catalog

    role = payload.get("_origin_role", "candidate")
    source = payload["ref_arch_src"] if role == "reference" else payload["kernel_src"]
    capture = Capture(source, candidate_windows_only=True) if payload["_origin_stage"] == "profile" else None
    catalog = Catalog() if capture else None
    state = _Calls(observer or capture, seeds, role, registry=capture.registry if capture else None)
    token = _ACTIVE.set(state)
    try:
        with (_source_guards(state, capture), tempfile.TemporaryDirectory(prefix="kb-origin-control-cache-") as cache,
              exclusive_compute(payload["eval_config"]) as exclusive):
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
        _ACTIVE.reset(token)
    if evaluator_digest() != payload["eval_config"].get("evaluator_sha256"):
        state.changes.append("evaluator_source")
    if not state.registry.intact():
        state.changes.append("registered_source")
    document = _document(payload, result, state, payload["_origin_stage"])
    document["exclusive_compute"] = exclusive
    if observer is not None:
        document["counter_windows"] = observer.windows
        document["pid"] = observer.pid
        document["counter_simulated"] = observer.simulated
    if capture:
        _finish_capture(capture, catalog, document, payload)
        document["raw"] = capture.raw
    return document


def validate_call_rows(rows, *, role="candidate"):
    """Backend-independent original callable, contents and state observations."""
    if (not isinstance(rows, list) or not rows or len(rows) > MAX_RECORDS
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("id") for row in rows] != list(range(1, len(rows) + 1))
            or any(row.get("role") != role
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
    return rows


def validate_call_timings(document, result, *, paired=False, baseline=False):
    """Bind original timer observations to the samples consumed by scoring."""
    timed = [row for row in document["calls"] if row["phase"] in {"warmup", "timing", "paired_warmup", "paired"}]
    timings = document["native_timings"]
    if (not isinstance(timings, list) or len(timings) != len(timed)
            or any(item.get("id") != row["id"] or item.get("native_call_sha256") != fingerprint(row)
                   or type(item.get("wall_ms")) not in (int, float) or not math.isfinite(item["wall_ms"]) or item["wall_ms"] <= 0
                   for item, row in zip(timings, timed))):
        raise ValueError("Native timer omitted or rebound a measured/warmup call")
    measured = [item["wall_ms"] for item, row in zip(timings, timed) if row["phase"] in {"timing", "paired"}]
    metadata = result["metadata"]
    if paired:
        key = "reference_ms" if document["binding"]["role"] == "reference" else "candidate_ms"
        if [pair[key] for pair in metadata["paired_timing"]["pairs"]] != measured:
            raise ValueError("Native paired samples differ from their observed candidate calls")
    elif timings:
        receipt = metadata["timing_receipt"]
        stats = result if baseline else result.get("runtime_stats", {})
        if (stats.get("samples_ms") != measured or receipt.get("first_timed_call_ms") != timings[0]["wall_ms"]
                or not math.isclose(receipt.get("warmup_total_ms", -1), sum(item["wall_ms"] for item in timings[:3]), rel_tol=1e-12)):
            raise ValueError("Native performance samples differ from their observed candidate calls")
    return timings


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
    rows = validate_call_rows(document["calls"], role=payload.get("_origin_role", "candidate"))
    if ([(row["phase"], row["seed"], row["label"], row["order"]) for row in rows] != expected
            or [row["id"] for row in rows] != list(range(1, len(expected) + 1))
            or document["seeds_consumed"] != len(seeds) or document["environment_changes"]):
        raise ValueError("Missing, changed, unverified or mode-unbound native candidate calls")
    validate_call_timings(document, result, paired=target == "paired_timing")
    if target == "evaluate" and seeds:
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
    request["eval_config"].setdefault("origin_minimum_custom_fraction", 0.95)
    request["eval_config"].setdefault("kernel_track", "custom")
    return request


def validate_scored(result, payload=None, *, allow_simulated=False):
    """Replay actual timer observations without granting execution ownership."""
    if result.get("metadata", {}).get("device_backend") == "neuron":
        from .neuron_runtime import validate_scored as native

        return native(result, payload, allow_simulated=allow_simulated)
    from ..evaluation.evaluate_task import validate_sources

    try:
        record = result["metadata"]["native_call_receipt"]
        request = record["native_request"]
        if (record.get("timed_callable_wrapped") is not False or type(record.get("execution_origin_collected")) is not bool
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
        origin = _observer_record(result, document["binding"])
        if (record.get("origin_capture_sha256") != origin["receipt_sha256"]
                or record["execution_origin_collected"] != (origin["status"] == "collected")):
            raise ValueError("Scored calls do not bind their actual origin observer")
        if record["execution_origin_collected"]:
            _bound_windows(origin["raw"], document, frozen, filtered=False)
        return frozen, document
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Missing or invalid actual scored native calls") from exc


def _observer_record(result, binding):
    origin = result.get("metadata", {}).get("origin_capture")
    if (not isinstance(origin, dict) or origin.get("protocol") != SCORED_CAPTURE
            or origin.get("mode") != SCORED_MODE or origin.get("binding") != binding
            or origin.get("receipt_sha256") != fingerprint({k: v for k, v in origin.items() if k != "receipt_sha256"})
            or origin.get("timing_sha256") != fingerprint(_scored_timing(result))):
        raise ValueError("Actual origin observer is missing, stale, changed or bound to other times")
    raw = origin.get("raw")
    if origin.get("status") == "not_measured":
        if raw is not None or not isinstance(origin.get("error"), str) or not origin["error"]:
            raise ValueError("Unavailable actual origin observer lacks its failure reason")
    elif (origin.get("status") != "collected" or origin.get("error") is not None or not isinstance(raw, dict)
          or raw.get("candidate_windows_only") is not False or raw.get("binding") != binding):
        raise ValueError("Actual origin requires continuous collection in the scoring process")
    return origin


def timing_observer(result, *, baseline=False, allow_simulated=False):
    """Verify the actual timing mode, without promoting source or work facts."""
    if result.get("metadata", {}).get("device_backend") == "neuron":
        from .neuron_runtime import timing_observer as native

        return native(result, baseline=baseline, allow_simulated=allow_simulated)
    from ..evaluation.evaluate_task import validate_sources

    try:
        metadata = result.get("metadata", {})
        if "native_call_receipt" in metadata and not baseline:
            _, document = validate_scored(result, allow_simulated=allow_simulated)
            origin = metadata["origin_capture"]
            device = document["binding"]["device_fingerprint"]
        else:
            origin = metadata["origin_capture"]
            request = origin["native_request"]
            validate_sources(request)
            _validate({**request, "task": "evaluate", "_origin_timing_seeds": origin["timing_seed_plan"]})
            device = result["device_fingerprint"]
            binding = {"native_request_sha256": fingerprint({**request, "_origin_timing_seeds": origin["timing_seed_plan"]}),
                       "role": "candidate", "evaluator_sha256": evaluator_digest(),
                       "device_fingerprint": device, "scope": BASELINE_SCOPE}
            _observer_record(result, binding)
            if (request.get("task") != "baseline" or request["eval_config"].get("evidence_level") not in {"L1", "L2"}
                    or request["eval_config"].get("diagnostic_stage")
                    or request["eval_config"].get("device_backend", "cuda") != "cuda"
                    or result.get("baseline_correctness") is not True
                    or result.get("reference_sha256") != fingerprint({"source": request["ref_arch_src"]})
                    or result.get("precision") != request["eval_config"]["precision"]
                    or result.get("timing_receipt", {}).get("input_seeds") != origin["timing_seed_plan"]
                    or origin.get("raw", {}).get("candidate_sha256") != hashlib.sha256(request["ref_arch_src"].encode()).hexdigest()
                    or origin.get("raw", {}).get("windows") != []):
                raise ValueError("Actual baseline observer has the wrong task or scope")
        raw = origin.get("raw")
        if (origin.get("status") != "collected" or not isinstance(raw, dict)
                or raw.get("protocol") != RAW_PROTOCOL or raw.get("environment_changes") != []
                or raw.get("capture_complete") is not True or raw.get("hooks_intact") is not True
                or raw.get("errors") != [] or type(raw.get("dropped_records")) is not int or raw["dropped_records"] != 0
                or type(raw.get("simulated")) is not bool
                or raw.get("simulated") is not False and not allow_simulated):
            raise ValueError("Actual timing observer was unavailable, incomplete, dropped records or simulated")
        if not allow_simulated:
            from ..analysis.calibration import identity

            identity(device)
        return SCORED_MODE
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Missing or invalid actual timing observer") from exc


def scored_origin(result, *, allow_simulated=False):
    """Replay source-to-launch facts from the actual timed execution itself."""
    if result.get("metadata", {}).get("device_backend") == "neuron":
        timing_observer(result, allow_simulated=allow_simulated)
        raise ValueError("Neuron NRT submissions are bound; device busy intervals and complete origin coverage are not yet available")
    timing_observer(result, allow_simulated=allow_simulated)
    request, _ = validate_scored(result, allow_simulated=allow_simulated)
    summary = summarize(result["metadata"]["origin_capture"]["raw"],
                        minimum_custom_fraction=request["eval_config"].get("origin_minimum_custom_fraction", 0.95))
    return {**summary, "scope": "actual_native_scored_calls", "native_timing_mode": SCORED_MODE,
            "timings_are_diagnostic": False, "formal_qualification_implemented": True,
            "evidence_scope": "scored_execution_origin_only; other_L1_L2_gates_required"}


def origin_check(result):
    """The scored path, rather than a diagnostic replay, supplies track evidence."""
    from ..evaluation.evidence import check

    try:
        summary = scored_origin(result)
        record = result["metadata"]["native_call_receipt"]
        track = record["native_request"]["eval_config"].get("kernel_track", "custom")
        if track not in {"custom", "library"}:
            raise ValueError("Unknown execution track")
        return check(summary["track_verdicts"][track], reason="Source-to-binary-to-launch attribution of actual timed execution",
                     values={"track": track, "mode": summary["native_timing_mode"], "raw_sha256": summary["raw_sha256"],
                             "native_call_receipt_sha256": record["receipt_sha256"],
                             "minimum_custom_fraction": summary["minimum_custom_fraction"],
                             "calls": [{**{key: row[key] for key in ("id", "unknown_activity_count",
                                          "custom_exclusive_fraction_lower_bound")},
                                        "status": row["track_verdicts"][track]} for row in summary["calls"]]})
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return check("not_measured", reason=f"Actual scored origin unavailable: {exc}"[:500])


def read_scored(path, problem, sample, task="evaluate", *, result_key=None):
    """Read one existing native artifact for both origin and counter previews."""
    if path is None:
        if result_key is not None:
            raise ValueError("--result-key requires a scored result map")
        return None
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Scored results must be an existing regular JSON file")
    with path.open("rb") as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("Scored results exceed the report byte limit")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("Scored results must contain a native result or result map")
    base, match = f"{problem}_{sample}", None
    if "metadata" not in result:
        key = result_key if result_key is not None else base + (":0" if task == "paired_timing" else "")
        suffix = r":([0-9]+)" if task == "paired_timing" else r"(?::([0-9]+):(standard|hidden))?"
        match = re.fullmatch(re.escape(base) + suffix, key) if isinstance(key, str) else None
        if match is None or key not in result:
            raise ValueError("Scored result key is missing or belongs to another slot/task")
        result = result[key]
    elif result_key is not None:
        raise ValueError("--result-key requires a result map, not an individual result")
    if not isinstance(result, dict) or not isinstance(result.get("metadata"), dict):
        raise ValueError("Selected scored result lacks native metadata")
    receipt = result["metadata"].get("native_call_receipt")
    request = receipt.get("native_request") if isinstance(receipt, dict) else None
    if not isinstance(request, dict) or not isinstance(request.get("eval_config"), dict):
        raise ValueError("Selected scored result lacks its native request")
    if match is not None:
        if request.get("task", "evaluate") != task:
            raise ValueError("Scored result task differs from its selected map key")
        repeat = match.group(1)
        settings = request.get("eval_config", {})
        if repeat is not None and (task != "paired_timing" or int(repeat) != 0 or "final_protocol" in settings):
            from ..evaluation.final_verification import PROTOCOL

            if (settings.get("final_protocol") != PROTOCOL or type(settings.get("final_repeat")) is not int
                    or settings["final_repeat"] != int(repeat)
                    or task == "evaluate" and bool(settings.get("use_hidden_tests")) != (match.group(2) == "hidden")):
                raise ValueError("Scored result phase/repeat differs from its selected map key")
    return result


def _bound_windows(raw, document, payload, *, filtered):
    rows = document["calls"]
    source = payload["ref_arch_src"] if payload.get("_origin_role") == "reference" else payload["kernel_src"]
    if (raw.get("candidate_sha256") != hashlib.sha256(source.encode()).hexdigest()
            or raw.get("candidate_windows_only") is not filtered or raw.get("binding") != document["binding"]
            or len(raw.get("windows", [])) != len(rows)
            or any(window.get("id") != row["id"] or window.get("native_call_sha256") != fingerprint(row)
                   or window.get("seed") != row["seed"] or window.get("output_verified") is not True
                   or fingerprint({"signature": window.get("input_signature")}) != row["input_signature_sha256"]
                   or any(window.get(key) != row[key] for key in ("training", "training_after", "no_grad"))
                   for window, row in zip(raw["windows"], rows))):
        raise ValueError("CUPTI windows do not bind every validated native candidate call")


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
    _bound_windows(raw, stages["profile"], payload, filtered=True)
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

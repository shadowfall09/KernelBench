"""Fresh-worker latency/burst contrast; SDK-light policy and receipt replay.

Both modes include identical output consumption and stable snapshot copies.
These diagnostic times never replace the benchmark's native timing samples.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import statistics
from contextlib import contextmanager
from contextvars import ContextVar

from .common import fingerprint
from .evidence import check

PROTOCOL = "kernelbench_pro_throughput_contrast_v1"
MODES = ("latency", "burst")
STAGES = {mode: f"throughput_{mode}" for mode in MODES}
THRESHOLD = 0.10
ACTIVE = ContextVar("kernelbench_throughput_audit", default=None)


def validate_inline(payload):
    settings = payload["eval_config"]
    if type(settings.get("throughput_validation", False)) is not bool:
        raise ValueError("Throughput validation requires an explicit boolean")
    if settings.get("throughput_validation") and any(name not in settings for name in (
            "throughput_calls", "throughput_repeats", "throughput_memory_mb")):
        raise ValueError("Missing resolved throughput call/repeat/memory policy")
    for name, default, low, high in (("throughput_calls", 8, 2, 128), ("throughput_repeats", 4, 4, 100),
                                      ("throughput_memory_mb", 1024, 1, 1048576)):
        value = settings.get(name, default)
        if type(value) is not int or not low <= value <= high or name == "throughput_repeats" and value % 2:
            raise ValueError("Invalid throughput call/repeat/memory policy")
    return (settings.get("throughput_validation", False) and not settings.get("use_hidden_tests", False)
            and payload.get("task", "evaluate") in {"evaluate", "paired_timing"})


def case_seed(nonce, repeat, slot):
    if not isinstance(nonce, str) or not re.fullmatch("[0-9a-f]{32}", nonce):
        raise ValueError("Throughput diagnostics require a parent-owned nonce")
    digest = hashlib.sha256(f"{nonce}/throughput/{repeat}/{slot}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def order_for(nonce, repeat):
    return "RC" if (int(nonce[-1], 16) + repeat) % 2 == 0 else "CR"


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def summarize(latency, burst):
    """Recompute differences, using each repeat's average milliseconds per call."""
    if (not isinstance(latency, list) or not isinstance(burst, list) or not latency or len(latency) != len(burst)
            or any(not positive(value) for value in [*latency, *burst])):
        raise ValueError("Throughput contrast requires complete positive paired samples")
    differences = [abs(a - b) / a for a, b in zip(latency, burst)]
    return {"latency_mean_ms": statistics.mean(latency), "burst_mean_ms": statistics.mean(burst),
            "relative_difference": abs(statistics.mean(latency) - statistics.mean(burst)) / statistics.mean(latency),
            "per_repeat_relative_difference": differences,
            "review_required": any(value > THRESHOLD and not math.isclose(value, THRESHOLD, rel_tol=0, abs_tol=1e-12)
                                   for value in differences), "threshold": THRESHOLD}


class OutputMismatch(ValueError):
    pass


def _schema(value):
    import torch

    if type(value) is torch.Tensor:
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError("Unsupported throughput tensor representation")
        count = value.numel()
        span = 1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride())) if count else 0
        if any(stride < 0 for stride in value.stride()):
            raise ValueError("Unsupported throughput tensor strides")
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype),
                "stride": list(value.stride()), "storage_bytes": max(count, span) * value.element_size()}
    if type(value) in (tuple, list):
        return {"kind": type(value).__name__, "items": [_schema(item) for item in value]}
    if type(value) is dict and all(type(key) in (str, int, bool) for key in value):
        return {"kind": "dict", "keys": list(value), "items": [_schema(item) for item in value.values()]}
    raise ValueError("Throughput outputs must be materialized tensor structures")


def _bytes(schema):
    return schema["storage_bytes"] if schema["kind"] == "tensor" else sum(_bytes(item) for item in schema["items"])


def _leaves(schema):
    return 1 if schema["kind"] == "tensor" else sum(_leaves(item) for item in schema["items"])


def _snapshot(schema, device):
    import torch

    if schema["kind"] == "tensor":
        return torch.empty(schema["shape"], dtype=getattr(torch, schema["dtype"].removeprefix("torch.")), device=device)
    items = [_snapshot(item, device) for item in schema["items"]]
    return dict(zip(schema["keys"], items)) if schema["kind"] == "dict" else tuple(items) if schema["kind"] == "tuple" else items


def _copy_output(destination, value):
    import torch

    if type(destination) is torch.Tensor:
        if (type(value) is not torch.Tensor or value.shape != destination.shape or value.dtype != destination.dtype
                or value.device != destination.device or value.layout != destination.layout):
            raise OutputMismatch("Timed output changed shape, dtype, device or representation")
        destination.copy_(value, non_blocking=True)
    elif type(destination) is not type(value) or len(destination) != len(value):
        raise OutputMismatch("Timed output changed structure")
    elif isinstance(destination, dict):
        if destination.keys() != value.keys():
            raise OutputMismatch("Timed output changed mapping keys")
        for key in destination:
            _copy_output(destination[key], value[key])
    else:
        for target, item in zip(destination, value):
            _copy_output(target, item)


class Collector:
    def __init__(self, payload, mode, *, clock=None, sync=None):
        settings = payload["eval_config"]
        if (not validate_inline(payload) or payload.get("task", "evaluate") != "evaluate" or mode not in MODES
                or settings.get("measure_performance") is not False or settings.get("diagnostic_stage") != STAGES.get(mode)):
            raise ValueError("Throughput control requires its own public correctness worker")
        from src.validation import clone_tensors, compare_outputs

        from .composite_oracles import meta_copy
        from .measurement import CLOCK, consume_output, synchronize

        self.payload, self.mode, self.row, self.factory = payload, mode, None, None
        self.clock, self.sync = clock or CLOCK, sync or synchronize
        self.consume, self.clone, self.compare, self.meta_copy = consume_output, clone_tensors, compare_outputs, meta_copy
        self.simulated = clock is not None or sync is not None
        self.native_changes = []

    def observe(self, reference, candidate, inputs, *, seed, device, set_seed, tolerance, label):
        if self.row is not None:
            return  # Never select an easier later case after a failed first one.
        import torch

        from .determinism import preserve_rng, signature
        from .input_calibration import verify_output
        from .measurement import input_signature

        settings = self.payload["eval_config"]
        count, repeats, nonce = settings["throughput_calls"], settings["throughput_repeats"], settings["diagnostic_nonce"]
        row = {"status": "not_measured", "label": label, "seed": seed, "input_signature": input_signature(inputs),
               "device": str(device), "tolerance": tolerance, "rounds": [], "warmup_calls": 0,
               "cpu_threads_before": torch.get_num_threads(),
               "training_before": {"R": getattr(reference, "training", None), "C": getattr(candidate, "training", None)},
               "estimated_storage_bytes": None, "output_schema": None, "environment_changes": []}
        self.row = row
        if self.factory is None or settings.get("device_backend", "cuda") == "cuda" and os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
            row["reason"] = "No trusted nominal factory, or launch blocking prevents an asynchronous contrast"
            return
        changes = row["environment_changes"]
        try:
            with torch.no_grad(), preserve_rng(device), settings_guard(changes):
                meta_args = [torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
                             if isinstance(value, torch.Tensor) else value for value in inputs]
                with torch.device("meta"):
                    schema = _schema(self.meta_copy(reference)(*meta_args))
                input_bytes = sum(_bytes(_schema(value)) for value in inputs if isinstance(value, torch.Tensor))
                output_bytes = _bytes(schema)
                # Three input sets (pristine/reference/candidate), two stable
                # output sets, bounded hash/copy temporaries and warmup storage.
                estimate = ((3 * count + 6) * input_bytes + (2 * count + 6) * output_bytes
                            + count * (4096 + 32 * _leaves(schema)))
                row.update(estimated_storage_bytes=estimate, output_schema=schema)
                if estimate > settings["throughput_memory_mb"] * 1024**2:
                    row["reason"] = "Original nominal diagnostic storage exceeds its envelope; no resizing or fewer calls"
                    return
                for slot in range(3):
                    warm_seed = case_seed(nonce, -1, slot)
                    set_seed(warm_seed)
                    args = self.factory()
                    if input_signature(args) != row["input_signature"]:
                        raise ValueError("Nominal input factory changed geometry or precision")
                    set_seed(warm_seed)
                    expected = self.clone(reference(*self.clone(args)))
                    self.sync(device)
                    set_seed(warm_seed)
                    with settings_guard(changes):
                        actual = candidate(*self.clone(args))
                        self.consume(actual)
                        self.sync(device)
                    if issue := self.compare(expected, actual, tolerance):
                        raise OutputMismatch(issue)
                    verify_output(reference, args, expected, actual)
                    row["warmup_calls"] += 1
                    del args, expected, actual
                for repeat in range(repeats):
                    base, hashes, seeds = [], [], [case_seed(nonce, repeat, slot) for slot in range(count)]
                    for input_seed in seeds:
                        set_seed(input_seed)
                        args = self.factory()
                        if input_signature(args) != row["input_signature"]:
                            raise ValueError("Nominal input factory changed geometry or precision")
                        hashes.append(fingerprint(signature(args)))
                        base.append(args)
                    prepared = {name: [self.clone(args) for args in base] for name in ("R", "C")}
                    saved = {name: [_snapshot(schema, device) for _ in range(count)] for name in ("R", "C")}
                    sequence_seed = case_seed(nonce, repeat, -1)
                    order, samples, drains, threads, no_grad, training = order_for(nonce, repeat), {}, {}, {}, {}, {}
                    models = {"R": reference, "C": candidate}
                    self.sync(device)
                    for name in order:
                        set_seed(sequence_seed)
                        self.sync(device)
                        call_times, consumed, observed_threads, observed_no_grad, observed_training = [], [], [], [], []
                        with settings_guard(changes):
                            started = self.clock() if self.mode == "burst" else None
                            for slot in range(count):
                                begin = self.clock() if self.mode == "latency" else None
                                output = models[name](*prepared[name][slot])
                                observed_no_grad.append(not torch.is_grad_enabled())
                                observed_training.append(getattr(models[name], "training", None))
                                if not observed_no_grad[-1] or observed_training[-1] != row["training_before"][name]:
                                    raise OutputMismatch("Forward changed no_grad or the declared model mode")
                                consumed.append(self.consume(output))
                                _copy_output(saved[name][slot], output)
                                observed_threads.append(torch.get_num_threads())
                                if self.mode == "latency":
                                    self.sync(device)
                                    call_times.append((self.clock() - begin) * 1000)
                            if self.mode == "burst":
                                self.sync(device)
                                elapsed = (self.clock() - started) * 1000
                            else:
                                elapsed = sum(call_times)
                        # A second barrier is diagnostic only, not evidence that
                        # arbitrary background threads can never enqueue later.
                        drain_start = self.clock()
                        self.sync(device)
                        drains[name] = max(0.0, (self.clock() - drain_start) * 1000)
                        samples[name] = {"total_ms": elapsed, "per_call_ms": elapsed / count, "call_samples_ms": call_times}
                        threads[name] = observed_threads
                        no_grad[name], training[name] = observed_no_grad, observed_training
                        del output, consumed
                    for index in range(count):
                        if issue := self.compare(saved["R"][index], saved["C"][index], tolerance):
                            raise OutputMismatch(issue)
                        verify_output(reference, base[index], saved["R"][index], saved["C"][index])
                    row["rounds"].append({"repeat": repeat, "order": order, "input_seeds": seeds, "input_sha256": hashes,
                        "sequence_seed": sequence_seed, "samples": samples, "post_barrier_ms": drains,
                        "cpu_threads_after_calls": threads, "outputs_verified": count,
                        "no_grad_after_calls": no_grad, "training_after_calls": training,
                        "output_sha256": {name: [fingerprint(signature(value)) for value in saved[name]] for name in ("R", "C")}})
                    del base, prepared, saved
                if changes:
                    raise OutputMismatch("Process settings changed during throughput diagnostics")
                row["status"] = "passed"
        except OutputMismatch as exc:
            row.update(status="failed", reason=str(exc))
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            row.update(status="not_measured", reason="Diagnostic metadata/storage/backend unavailable: " + type(exc).__name__)

    def document(self):
        row = copy.deepcopy(self.row)
        if row is not None and self.native_changes:
            row["environment_changes"] = sorted(set([*row["environment_changes"], *self.native_changes]))
            row.update(status="failed", reason="Candidate changed settings during native import/construction/calls")
        return {"protocol": PROTOCOL, "mode": self.mode, "request_sha256": fingerprint(self.payload),
                "calls": self.payload["eval_config"]["throughput_calls"], "repeats": self.payload["eval_config"]["throughput_repeats"],
                "memory_mb": self.payload["eval_config"]["throughput_memory_mb"], "case": row,
                "simulated_execution": self.simulated, "timings_are_diagnostic": True, "scored_wall_times_replaced": False,
                "output_snapshot_policy": "preallocated_current_stream_copy_inside_both_modes",
                "model_state_policy": "fresh_worker_per_mode; equal_reference_candidate_call_counts; no_resets_between_calls",
                "complete_gpu_integrity_proven": False}


@contextmanager
def audit(payload, mode):
    collector = Collector(payload, mode)
    token = ACTIVE.set(collector)
    try:
        yield collector
    finally:
        ACTIVE.reset(token)


def bind_factory(factory):
    if (collector := ACTIVE.get()) is not None:
        collector.factory = factory


@contextmanager
def settings_guard(changes):
    import torch

    from .strict_validation import environment_guard

    grad = torch.is_grad_enabled()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        with environment_guard(changes):
            yield
    finally:
        if torch.is_grad_enabled() != grad:
            changes.append("grad_enabled")
            torch.set_grad_enabled(grad)
        if (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.utils.deterministic.fill_uninitialized_memory) != (deterministic, warn, fill):
            changes.append("deterministic_policy")
            torch.use_deterministic_algorithms(deterministic, warn_only=warn)
            torch.utils.deterministic.fill_uninitialized_memory = fill


@contextmanager
def candidate_guard():
    if (collector := ACTIVE.get()) is None:
        yield
    else:
        with settings_guard(collector.native_changes):
            yield


def observe(*args, **kwargs):
    if (collector := ACTIVE.get()) is not None:
        collector.observe(*args, **kwargs)


DTYPE_BYTES = {**dict.fromkeys(("torch.bool", "torch.uint8", "torch.int8", "torch.float8_e4m3fn", "torch.float8_e4m3fnuz",
                               "torch.float8_e5m2", "torch.float8_e5m2fnuz", "torch.float8_e8m0fnu"), 1),
               **dict.fromkeys(("torch.int16", "torch.uint16", "torch.float16", "torch.bfloat16"), 2),
               **dict.fromkeys(("torch.int32", "torch.uint32", "torch.float32", "torch.complex32"), 4),
               **dict.fromkeys(("torch.int64", "torch.uint64", "torch.float64", "torch.complex64"), 8), "torch.complex128": 16}


def _storage_bytes(item):
    shape, stride, dtype = item["shape"], item["stride"], item["dtype"]
    if (not isinstance(shape, list) or not isinstance(stride, list) or len(shape) != len(stride)
            or any(type(n) is not int or n < 0 for n in shape + stride) or dtype not in DTYPE_BYTES):
        raise ValueError("Invalid throughput geometry or storage dtype")
    count = math.prod(shape)
    span = 1 + sum((n - 1) * step for n, step in zip(shape, stride)) if count else 0
    return max(count, span) * DTYPE_BYTES[dtype]


def _schema_bytes(item):
    if not isinstance(item, dict):
        raise ValueError("Invalid throughput output schema")
    if item.get("kind") == "tensor":
        size = _storage_bytes(item)
        if type(item.get("storage_bytes")) is not int or item["storage_bytes"] != size:
            raise ValueError("Output storage estimate changed")
        return size
    children = item.get("items")
    if item.get("kind") not in {"dict", "tuple", "list"} or not isinstance(children, list) or not children:
        raise ValueError("Invalid throughput output container")
    if item["kind"] == "dict" and (not isinstance(item.get("keys"), list) or len(item["keys"]) != len(children)
                                    or len(set(item["keys"])) != len(children)):
        raise ValueError("Invalid throughput output mapping")
    return sum(_schema_bytes(child) for child in children)


def verdict(documents, requests, cases, *, allow_simulated=False):
    """Recompute policy, call coverage, storage admission and every timing flag."""
    def missing(reason):
        return check("not_measured", reason=reason)

    rows = {}
    try:
        for mode in MODES:
            doc, request, native = documents[mode], requests[mode], cases[mode]
            settings = request["eval_config"]
            if (not isinstance(doc, dict) or doc.get("protocol") != PROTOCOL or doc.get("mode") != mode
                    or doc.get("request_sha256") != fingerprint(request) or not validate_inline(request)
                    or doc.get("calls") != settings["throughput_calls"] or doc.get("repeats") != settings["throughput_repeats"]
                    or doc.get("memory_mb") != settings["throughput_memory_mb"]
                    or doc.get("timings_are_diagnostic") is not True or doc.get("scored_wall_times_replaced") is not False
                    or doc.get("complete_gpu_integrity_proven") is not False
                    or type(doc.get("simulated_execution")) is not bool or doc["simulated_execution"] and not allow_simulated
                    or doc.get("output_snapshot_policy") != "preallocated_current_stream_copy_inside_both_modes"
                    or doc.get("model_state_policy") != "fresh_worker_per_mode; equal_reference_candidate_call_counts; no_resets_between_calls"):
                return missing("No matching fresh-worker throughput policy or source binding")
            row = doc.get("case")
            if not isinstance(row, dict) or not isinstance(native, list) or not native or not isinstance(native[0], dict):
                return missing("Throughput omitted the first actual public native case")
            if (row.get("label"), row.get("seed"), row.get("input_signature")) != (
                    native[0].get("label"), native[0].get("seed"), native[0].get("input_signature")):
                return missing("Throughput changed its actual nominal input/case/seed binding")
            if not allow_simulated and str(row.get("device", "")).split(":")[0] != settings.get("device_backend", "cuda"):
                return missing("CPU/simulated throughput cannot qualify a device")
            if row.get("status") == "failed":
                return check("failed", reason="Native outputs or settings failed the snapshot-consumer throughput pipeline")
            if row.get("status") != "passed" or row.get("environment_changes") != [] or row.get("warmup_calls") != 3:
                return missing("Throughput did not complete its warmup, backend or settings checks")
            count, repeats, nonce = settings["throughput_calls"], settings["throughput_repeats"], settings["diagnostic_nonce"]
            expected_tolerance = settings["fp32_tolerance"] if settings["precision"] == "fp32" or settings.get("device_backend") == "neuron" else 1e-2
            if type(row.get("tolerance")) not in (int, float) or row["tolerance"] != expected_tolerance:
                return missing("Throughput comparison policy changed")
            input_bytes = sum(_storage_bytes(item) for item in row["input_signature"] if "shape" in item)
            output_bytes = _schema_bytes(row["output_schema"])
            estimate = ((3 * count + 6) * input_bytes + (2 * count + 6) * output_bytes
                        + count * (4096 + 32 * _leaves(row["output_schema"])))
            if (type(row.get("estimated_storage_bytes")) is not int or row["estimated_storage_bytes"] != estimate
                    or estimate > settings["throughput_memory_mb"] * 1024**2
                    or type(row.get("cpu_threads_before")) is not int or row["cpu_threads_before"] < 1
                    or not isinstance(row.get("training_before"), dict)
                    or set(row["training_before"]) != {"R", "C"}
                    or any(type(value) not in (bool, type(None)) for value in row["training_before"].values())):
                return missing("Throughput memory admission or thread identity changed")
            rounds = row.get("rounds")
            if not isinstance(rounds, list) or len(rounds) != repeats:
                return missing("Throughput omitted declared repeats")
            for repeat, record in enumerate(rounds):
                if (not isinstance(record, dict) or record.get("repeat") != repeat or record.get("order") != order_for(nonce, repeat)
                        or record.get("input_seeds") != [case_seed(nonce, repeat, slot) for slot in range(count)]
                        or record.get("sequence_seed") != case_seed(nonce, repeat, -1) or record.get("outputs_verified") != count):
                    return missing("Throughput changed its balanced order, seeds or output count")
                digests = [*record.get("input_sha256", []), *record.get("output_sha256", {}).get("R", []),
                           *record.get("output_sha256", {}).get("C", [])]
                if len(digests) != 3 * count or any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in digests):
                    return missing("Throughput omitted input/output byte snapshots")
                for name in ("R", "C"):
                    sample = record["samples"][name]
                    parts = sample["call_samples_ms"]
                    if (not positive(sample["total_ms"]) or not positive(sample["per_call_ms"])
                            or not math.isclose(sample["total_ms"] / count, sample["per_call_ms"], rel_tol=1e-12)
                            or not isinstance(parts, list) or mode == "burst" and parts != []
                            or mode == "latency" and (len(parts) != count or any(not positive(part) for part in parts)
                                or not math.isclose(sum(parts), sample["total_ms"], rel_tol=1e-12))
                            or type(record["post_barrier_ms"][name]) not in (int, float)
                            or not math.isfinite(record["post_barrier_ms"][name]) or record["post_barrier_ms"][name] < 0
                            or record["cpu_threads_after_calls"][name] != [row["cpu_threads_before"]] * count
                            or record["no_grad_after_calls"][name] != [True] * count
                            or record["training_after_calls"][name] != [row["training_before"][name]] * count):
                        return missing("Throughput samples, per-call thread counts or barriers are incomplete")
            rows[mode] = row
        first, second = rows["latency"], rows["burst"]
        if any(first[key] != second[key] for key in ("label", "seed", "input_signature", "output_schema", "cpu_threads_before", "training_before", "estimated_storage_bytes")):
            return missing("Fresh throughput workers did not measure the same workload")
        if any(a["input_sha256"] != b["input_sha256"] for a, b in zip(first["rounds"], second["rounds"])):
            return missing("Fresh throughput workers did not replay identical input contents")
        summaries = {name: summarize([record["samples"][name]["per_call_ms"] for record in first["rounds"]],
                                     [record["samples"][name]["per_call_ms"] for record in second["rounds"]]) for name in ("R", "C")}
        return check("passed", reason="Both fresh modes validated every snapshot; discrepancies remain timing-review flags",
            values={"protocol": PROTOCOL, "models": summaries, "review_required": any(row["review_required"] for row in summaries.values()),
                    "threshold": THRESHOLD, "calls": requests["latency"]["eval_config"]["throughput_calls"],
                    "repeats": requests["latency"]["eval_config"]["throughput_repeats"],
                    "scope": "public_nominal_snapshot_consumer_pipeline; diagnostic_only", "complete_gpu_integrity_proven": False})
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return missing("Malformed or incomplete throughput evidence")

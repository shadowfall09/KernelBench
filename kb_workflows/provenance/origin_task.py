"""Native call coverage and separate control/profile origin diagnostics."""

from __future__ import annotations

import copy
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
from .kernel_origin import MAX_RECORDS, summarize

NATIVE_PROTOCOL = "kernelbench_pro_native_origin_v1"
_ACTIVE = ContextVar("kernelbench_native_origin_calls", default=None)
MAX_REPORT_BYTES = 64 * 1024**2


def _validate(payload):
    settings = payload["eval_config"]
    target = payload.get("_origin_target", payload.get("task", "evaluate"))
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


class _Calls:
    def __init__(self, capture, seeds):
        self.capture, self.seeds = capture, seeds
        self.rows, self.windows, self.changes = [], {}, []
        self.cursor, self.last_seed = 0, None

    def next_seed(self):
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
        row = {"id": len(self.rows) + 1, "phase": phase, "seed": seed, "label": label, "order": order,
               "input_content_sha256": fingerprint({"value": signature(args, typed=True)}),
               "input_signature_sha256": fingerprint({"signature": geometry}),
               "input_layout_sha256": fingerprint(aliases),
               "called": False, "verified": False, "output_content_sha256": None}
        self.rows.append(row)

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
def candidate_call(model, args, **scope):
    """Inactive calls retain the exact original callable, with no timed wrapper."""
    state = _ACTIVE.get()
    yield state.call(model, args, **scope) if state is not None else (model, None)


def snapshot_output(row, output):
    if row is not None:
        from ..evaluation.determinism import signature

        row["output_content_sha256"] = fingerprint({"value": signature(output, typed=True)})


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
            return function(payload, *args, **kwargs)
        if stage not in {"control", "profile"}:
            raise ValueError("Unknown native origin worker stage")
        return _run_native(payload, lambda: function(payload, *args, **kwargs))
    return wrapped


def _run_native(payload, invoke):
    _, seeds, _ = _validate(payload)
    from src import eval as native

    from ..evaluation.strict_validation import environment_guard
    from .cuda_origin import Capture
    from .origin_binaries import Catalog

    capture = Capture(payload["kernel_src"], candidate_windows_only=True) if payload["_origin_stage"] == "profile" else None
    catalog = Catalog() if capture else None
    state = _Calls(capture, seeds)
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
                    with capture.recording() if capture else nullcontext():
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
    canonical = copy.deepcopy(payload)
    canonical.pop("_origin_stage")
    binding = {"native_request_sha256": fingerprint(canonical),
               "evaluator_sha256": payload["eval_config"].get("evaluator_sha256"),
               "device_fingerprint": result.get("metadata", {}).get("device_fingerprint"),
               "scope": "native_correctness_both_orders_warmups_timing_and_paired; memory_probe_and_other_diagnostics_excluded"}
    document = {"protocol": NATIVE_PROTOCOL, "stage": payload["_origin_stage"], "binding": binding,
                "native_result": result, "calls": state.rows, "seeds_consumed": state.cursor,
                "environment_changes": sorted(set(state.changes))}
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
            or document["seeds_consumed"] != len(seeds) or document["environment_changes"]
            or any(not row["called"] or not row["verified"] or not row.get("no_grad")
                   or row.get("training_after") != row.get("training") or not row.get("output_content_sha256") for row in rows)):
        raise ValueError("Missing, changed, unverified or mode-unbound native candidate calls")
    return rows


def audit(payload, *, run=None):
    from ..evaluation.evaluate_task import validate_sources
    from ..evaluation.tool_runner import environment, process

    target, _, threshold = _validate(payload)
    validate_sources(payload)
    if run is None:
        from importlib.metadata import version

        from .cuda_origin import abi_supported

        if (not abi_supported() or any(version(package).split("+")[0] != expected for package, expected in (
                ("cupti-python", "13.3.0"), ("triton", "3.7.0"), ("torch", "2.12.0")))):
            raise ValueError("Native origin requires its pinned CUPTI/Triton/Torch packages and Linux ABI")
    limit = (payload["eval_config"]["timeout"] - 10) / 2
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Origin control/profile need positive budgets including process cleanup")
    runner, stages = run or process, {}
    deadline = time.monotonic() + payload["eval_config"]["timeout"]

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
            canonical = copy.deepcopy(child)
            canonical.pop("_origin_stage")
            if (not isinstance(document, dict) or document.get("protocol") != NATIVE_PROTOCOL or document.get("stage") != stage
                    or not isinstance(document.get("binding"), dict)
                    or document.get("binding", {}).get("native_request_sha256") != fingerprint(canonical)):
                stages[stage] = document
                return failed("Native origin worker changed its request binding or stage", directory)
            document["worker_wall_seconds"] = outcome.get("request_wall_seconds")
            stages[stage] = document
        if not isinstance(stages["profile"].get("raw"), dict):
            return failed("Profile worker omitted its raw source/launch capture", root / "profile")
    raw = stages["profile"].pop("raw")
    try:
        control, profiled = (_coverage(stages[stage], {**payload, "task": target}) for stage in ("control", "profile"))
        if control != profiled or stages["control"]["binding"] != stages["profile"]["binding"]:
            raise ValueError("Control/profile native input, output, mode observations or device differ")
        if (raw.get("candidate_windows_only") is not True or raw.get("binding") != stages["profile"]["binding"]
                or len(raw.get("windows", [])) != len(profiled)
                or any(window.get("id") != row["id"] or window.get("native_call_sha256") != fingerprint(row)
                       or window.get("seed") != row["seed"] or window.get("output_verified") is not True
                       or fingerprint({"signature": window.get("input_signature")}) != row["input_signature_sha256"]
                       or any(window.get(key) != row[key] for key in ("training", "training_after", "no_grad"))
                       for window, row in zip(raw["windows"], profiled))):
            raise ValueError("CUPTI windows do not bind every validated native candidate call")
        coverage = {"status": "passed", "candidate_calls": len(control), "bitwise_observable_replay": True,
                    "simulated": run is not None, "memory_probe_included": False, "uninstrumented_path_equivalence_proven": False}
        summary = summarize(raw, minimum_custom_fraction=threshold)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        coverage = {"status": "not_measured", "reason": str(exc)}
        summary = {"status": "not_measured", "reason": str(exc), "timings_are_diagnostic": True,
                   "formal_qualification_implemented": False}
    return {"status": "ok", "purpose": "native_execution_origin_diagnostics_only", "raw": raw,
            "native_runs": stages, "native_coverage": coverage, "summary": summary,
            "scored_wall_times_replaced": False, "formal_qualification_implemented": False}

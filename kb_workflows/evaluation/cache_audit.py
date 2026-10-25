"""Source-bound same-object input updates in a fresh correctness worker.

This is functional cache validation, not a proof about arbitrary GPU activity.
SDK imports remain inside worker helpers so orchestration stays SDK-light.
"""

from __future__ import annotations

import copy
import hashlib
import re
from contextlib import contextmanager
from contextvars import ContextVar

from ..common import fingerprint
from .cache_recipes import plan
from .evidence import check

PROTOCOL = "kernelbench_pro_input_cache_v1"
STAGE = "input_cache"
ACTIVE = ContextVar("kernelbench_input_cache", default=None)


def validate_inline(payload):
    settings = payload["eval_config"]
    enabled = settings.get("cache_validation", False)
    memory = settings.get("cache_memory_mb", 1024)
    if type(enabled) is not bool or type(memory) is not int or not 1 <= memory <= 1048576:
        raise ValueError("Cache validation requires an explicit boolean and positive memory envelope")
    if enabled and "cache_memory_mb" not in settings:
        raise ValueError("Missing resolved input cache memory policy")
    if enabled and "ref_arch_src" in payload and settings.get("input_cache_plan") != plan(
            payload["level"], payload["problem_id"], payload["ref_arch_src"]):
        raise ValueError("Missing or stale source-bound input cache plan")
    return enabled and payload.get("task", "evaluate") in {"evaluate", "paired_timing"}


def probe_seed(nonce, label, seed, probe):
    if not isinstance(nonce, str) or not re.fullmatch("[0-9a-f]{32}", nonce):
        raise ValueError("Input cache diagnostics require a parent-owned nonce")
    return int.from_bytes(hashlib.sha256(f"{nonce}/cache/{label}/{seed}/{probe}".encode()).digest()[:4], "big")


def estimate_bytes(storage, logical, output, public):
    # Full backing stores for the two isolated aliases, bounded factory/recipe,
    # hashing/copy temporaries, current/previous output snapshots. Does not bound
    # model weights, candidate workspaces or the Python/OS process heap.
    return 2 * storage + (32 if public else 12) * logical + 6 * output


class OutputMismatch(ValueError):
    pass


class Collector:
    def __init__(self, payload, *, sync=None):
        settings = payload["eval_config"]
        if (not validate_inline(payload) or payload.get("task") != "evaluate"
                or settings.get("measure_performance") is not False or settings.get("diagnostic_stage") != STAGE):
            raise ValueError("Input cache validation requires its separate correctness worker")
        from src.validation import clone_tensors, compare_outputs

        from ..numerics.input_calibration import verify_output
        from .determinism import signature
        from .input_reuse import MutableInputs, description, isolated, logical_bytes, storage_bytes
        from .measurement import synchronize

        self.payload, self.rows, self.factory, self.native_changes = payload, [], None, []
        self.sync, self.simulated = sync or synchronize, sync is not None
        self.clone, self.compare, self.signature, self.verify = clone_tensors, compare_outputs, signature, verify_output
        self.mutable, self.description, self.isolated = MutableInputs, description, isolated
        self.logical_bytes, self.storage_bytes = logical_bytes, storage_bytes

    def observe(self, reference, candidate, inputs, *, reference_output, seed, tolerance, device, set_seed, label):
        import torch

        from .cache_recipes import change
        from .determinism import preserve_rng
        from .measurement import input_signature
        from .throughput_audit import _bytes, _schema, settings_guard

        settings = self.payload["eval_config"]
        public = not settings.get("use_hidden_tests", False)
        row = {"label": label, "seed": seed, "input_signature": input_signature(inputs), "status": "not_measured",
               "device": str(device), "tolerance": tolerance, "probes": [], "environment_changes": [],
               "value_policy": "trusted_nominal_factory" if public else "reviewed_value_only_recipe"}
        self.rows.append(row)
        try:
            if public and self.factory is None:
                raise ValueError("No trusted nominal input factory")
            schema, stores, logical = _schema(reference_output), self.storage_bytes(inputs), self.logical_bytes(inputs)
            estimate = estimate_bytes(stores, logical, _bytes(schema), public)
            row.update(output_schema=schema, input_storage_bytes=stores, input_logical_bytes=logical,
                       estimated_storage_bytes=estimate, input_aliases=self.description(inputs))
            if estimate > settings["cache_memory_mb"] * 1024**2:
                raise ValueError("Original diagnostic storage exceeds its envelope; no resizing or fewer probes")
            with torch.no_grad(), preserve_rng(device):
                groups = {name: self.mutable(inputs) for name in ("R", "C")}
                models, previous = {"R": reference, "C": candidate}, None
                modes = {name: getattr(model, "training", None) for name, model in models.items()}
                row["training_before"] = modes
                for probe in range(3):
                    value_seed = probe_seed(settings["diagnostic_nonce"], label, seed, probe)
                    set_seed(value_seed)
                    pristine = inputs if probe == 0 else self.factory() if public else change(
                        self.isolated(inputs), settings["input_cache_plan"], probe=probe, reference=reference)
                    if (self.description(pristine) != row["input_aliases"] or self.storage_bytes(pristine) > stores
                            or self.logical_bytes(pristine) != logical):
                        raise ValueError("Probe changed input geometry, storage envelope, aliases or declared roles")
                    updates = {name: group.update(pristine) for name, group in groups.items()}
                    digest = fingerprint(self.signature(pristine))
                    order, outputs = ("CR" if probe % 2 == 0 else "RC"), {}
                    for name in order:
                        set_seed(value_seed)
                        self.sync(device)
                        with settings_guard(row["environment_changes"]):
                            try:
                                output = models[name](*groups[name].values)
                            except Exception as exc:
                                if name == "C":
                                    raise OutputMismatch("Candidate failed under an input update: " + type(exc).__name__) from exc
                                raise
                            self.sync(device)
                            outputs[name] = self.clone(output)
                            self.sync(device)
                        if getattr(models[name], "training", None) != modes[name]:
                            raise OutputMismatch("Forward changed the declared model mode")
                    if issue := self.compare(outputs["R"], outputs["C"], tolerance):
                        raise OutputMismatch(issue)
                    self.verify(reference, pristine, outputs["R"], outputs["C"])
                    hashes = {name: fingerprint(self.signature(value)) for name, value in outputs.items()}
                    row["probes"].append({"probe": probe, "seed": value_seed, "order": order, "input_sha256": digest,
                        "output_sha256": hashes, "updates": updates, "outputs_verified": True,
                        "reference_output_changed": None if previous is None else hashes["R"] != previous})
                    previous = hashes["R"]
                    del outputs, pristine, output
                if row["environment_changes"]:
                    raise OutputMismatch("Process settings changed during input cache diagnostics")
                row["status"] = "passed"
        except OutputMismatch as exc:
            row.update(status="failed", reason=str(exc))
        except Exception as exc:
            row.update(status="not_measured", reason="Input/cache adapter unavailable: " + type(exc).__name__ + ": " + str(exc))

    def document(self):
        return {"protocol": PROTOCOL, "request_sha256": fingerprint(self.payload), "cases": copy.deepcopy(self.rows),
                "memory_mb": self.payload["eval_config"]["cache_memory_mb"], "simulated_execution": self.simulated,
                "native_environment_changes": sorted(set(self.native_changes)), "probe_count": 3,
                "model_state_policy": "equal_calls; no_resets_between_probes", "scored_wall_times_replaced": False}


@contextmanager
def audit(payload):
    collector = Collector(payload)
    token = ACTIVE.set(collector)
    try:
        yield collector
    finally:
        ACTIVE.reset(token)


def bind_factory(factory):
    if (collector := ACTIVE.get()) is not None:
        collector.factory = factory


@contextmanager
def candidate_guard():
    if (collector := ACTIVE.get()) is None:
        yield
    else:
        from .throughput_audit import settings_guard

        with settings_guard(collector.native_changes):
            yield


def observe(*args, **kwargs):
    if (collector := ACTIVE.get()) is not None:
        collector.observe(*args, **kwargs)


def verdict(document, request, native_cases, *, allow_simulated=False):
    def missing(reason):
        return check("not_measured", reason=reason)

    try:
        settings = request["eval_config"]
        if (not validate_inline(request) or not isinstance(document, dict) or document.get("protocol") != PROTOCOL
                or document.get("request_sha256") != fingerprint(request) or document.get("probe_count") != 3
                or document.get("memory_mb") != settings["cache_memory_mb"]
                or type(document.get("simulated_execution")) is not bool or document["simulated_execution"] and not allow_simulated
                or document.get("scored_wall_times_replaced") is not False
                or document.get("model_state_policy") != "equal_calls; no_resets_between_probes"):
            return missing("No matching source-bound same-object input cache policy")
        rows = document.get("cases")
        if (not isinstance(rows, list) or not rows or not isinstance(native_cases, list) or not native_cases
                or [(r["label"], r["seed"], r["input_signature"]) for r in rows]
                   != [(r["label"], r["seed"], r["input_signature"]) for r in native_cases if r.get("passed") is True]):
            return missing("Input cache audit omitted or changed native correctness cases")
        if document.get("native_environment_changes"):
            return check("failed", reason="Candidate changed process settings during import, construction or native calls")
        if document.get("native_environment_changes") != []:
            return missing("Input cache audit omitted native settings observations")
        from .throughput_audit import _schema_bytes, _storage_bytes

        for row in rows:
            if not allow_simulated and str(row["device"]).split(":")[0] != settings.get("device_backend", "cuda"):
                return missing("CPU/simulated input updates cannot qualify a device")
            if row.get("status") == "failed":
                return check("failed", reason="Same-object input update produced wrong outputs or changed settings")
            public = not settings.get("use_hidden_tests", False)
            expected_tolerance = settings["fp32_tolerance"] if settings["precision"] == "fp32" or settings.get("device_backend") == "neuron" else 1e-2
            if (row.get("status") != "passed" or row.get("environment_changes") != [] or row.get("tolerance") != expected_tolerance
                    or row.get("value_policy") != ("trusted_nominal_factory" if public else "reviewed_value_only_recipe")):
                return missing("Input cache audit did not complete its declared value/comparison policy")
            logical = sum(_storage_bytes({**item, "stride": [0] * len(item["shape"])}) for item in row["input_signature"] if "shape" in item)
            stores = row["input_storage_bytes"]
            if type(stores) is not int or stores < 0 or row.get("input_logical_bytes") != logical:
                return missing("Input cache storage accounting changed")
            aliases = row["input_aliases"]
            shapes = [item for item in row["input_signature"] if "shape" in item]
            if (not isinstance(aliases, list) or len(aliases) != len(shapes)
                    or any(any(alias.get(key) != item.get(key) for key in ("shape", "stride", "dtype"))
                           for alias, item in zip(aliases, shapes))):
                return missing("Input alias geometry differs from native inputs")
            objects, groups, minimum = {}, {}, {}
            for alias in aliases:
                obj, store, offset = alias["object"], alias["storage"], alias["offset"]
                if any(type(value) is not int or value < 0 for value in (obj, store, offset)):
                    return missing("Invalid input alias indices")
                if obj != objects.setdefault(obj, len(objects)) or store != groups.setdefault(store, len(groups)):
                    return missing("Input alias indices are not canonical")
                if alias.get("device") != row["device"]:
                    return missing("Input alias device changed")
                from .throughput_audit import DTYPE_BYTES

                count = 1
                for size in alias["shape"]:
                    count *= size
                span = ((offset + 1 + sum((size - 1) * stride for size, stride in zip(alias["shape"], alias["stride"])))
                        * DTYPE_BYTES[alias["dtype"]] if count else 0)
                minimum[store] = max(span, minimum.get(store, 0))
            if sum(minimum.values()) > stores:
                return missing("Backing storage does not contain declared views")
            estimate = estimate_bytes(stores, logical, _schema_bytes(row["output_schema"]), public)
            if (type(row.get("estimated_storage_bytes")) is not int or row["estimated_storage_bytes"] != estimate
                    or estimate > settings["cache_memory_mb"] * 1024**2):
                return missing("Input cache memory admission changed")
            probes = row.get("probes")
            if not isinstance(probes, list) or len(probes) != 3:
                return missing("Input cache audit omitted declared probes")
            previous = None
            for index, probe in enumerate(probes):
                if (probe.get("probe") != index or probe.get("seed") != probe_seed(settings["diagnostic_nonce"], row["label"], row["seed"], index)
                        or probe.get("order") != ("CR" if index % 2 == 0 else "RC") or probe.get("outputs_verified") is not True):
                    return missing("Input cache probe changed seeds, order or output verification")
                hashes = probe["output_sha256"]
                if (set(hashes) != {"R", "C"} or any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
                        for value in [probe["input_sha256"], *hashes.values()])
                        or probe.get("reference_output_changed") != (None if previous is None else hashes["R"] != previous)):
                    return missing("Input cache audit omitted byte snapshots")
                previous = hashes["R"]
                updates = probe["updates"]
                if set(updates) != {"R", "C"} or any(update.get("same_tensor_objects") is not True
                        or type(update.get("storage_preserved_during_update")) is not bool
                        or type(update.get("geometry_restorations")) is not int or update["geometry_restorations"] < 0
                        for update in updates.values()):
                    return missing("Input cache update did not retain tensor objects")
        return check("passed", reason="Every native case verified three same-object input updates",
                     values={"case_count": len(rows), "probes_per_case": 3, "scope": "functional_consistency; no_gpu_integrity_proof"})
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return missing("Malformed or incomplete same-object input cache evidence")

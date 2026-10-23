"""State equivalence for registered and source-bound ordinary tensor state.

This is a functional audit, not a sandbox for untrusted Python. The mapping only
locates candidate state; it cannot supply expected values or comparison code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass

import torch

from src.validation import clone_tensors, compare_outputs, synchronize

from .evidence import check
from .state_audit import MODE_SEQUENCE
from .state_audit import STATE_PROTOCOL as PROTOCOL
from .state_paths import plain_tensors

MAP_ATTRIBUTE = "kernelbench_state_map"
MAX_STATES = 100_000


class StateContractError(ValueError):
    """A representation cannot satisfy the declared state contract."""


def _registered(model):
    """Include tied names and non-persistent buffers; avoid state_dict overrides."""
    states, modules = {}, set()

    def walk(module, prefix, ancestors):
        if not isinstance(module, torch.nn.Module) or id(module) in ancestors:
            raise StateContractError("Invalid or cyclic registered module graph")
        modules.add(id(module))
        namespace = vars(module)
        for attribute, kind in (("_parameters", "parameter"), ("_buffers", "buffer")):
            entries = namespace.get(attribute)
            if type(entries) is not dict:
                raise StateContractError("Registered state must use ordinary module registries")
            for name, tensor in entries.items():
                if tensor is None:
                    continue
                if (not isinstance(name, str) or not name or "." in name
                        or type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                        or tensor.layout != torch.strided or tensor.is_quantized
                        or tensor.device.type == "meta"):
                    raise StateContractError("State must be a materialized dense registered tensor")
                states[prefix + name] = (kind, tensor)
                if len(states) > MAX_STATES:
                    raise StateContractError("State contract exceeds its tensor limit")
        children = namespace.get("_modules")
        if type(children) is not dict:
            raise StateContractError("Registered modules must use an ordinary registry")
        for name, child in children.items():
            if child is not None:
                if not isinstance(name, str) or not name or "." in name:
                    raise StateContractError("Invalid registered module name")
                walk(child, prefix + name + ".", ancestors | {id(module)})

    walk(model, "", set())
    return states, modules


def _plain_tensors(model):
    """Find ordinary tensor attributes/containers without invoking user iterators."""
    result, seen = {}, set()

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            result[path] = value
        elif id(value) not in seen:
            seen.add(id(value))
            if isinstance(value, torch.nn.Module):
                for name, item in vars(value).items():
                    if name not in {"_parameters", "_buffers", "_modules"}:
                        walk(item, path + "." + name)
                for name, item in vars(value)["_modules"].items():
                    walk(item, path + "." + name)
            elif type(value) is dict:
                for index, item in enumerate(value.values()):
                    walk(item, path + f"[{index}]")
            elif type(value) in (tuple, list):
                for index, item in enumerate(value):
                    walk(item, path + f"[{index}]")
        if len(result) > MAX_STATES:
            raise StateContractError("Plain tensor state exceeds its audit limit")

    walk(model, "model")
    return result


def _declaration(model):
    """Read data, never execute a mapping method/property supplied by a candidate."""
    namespace = vars(model)
    if MAP_ATTRIBUTE in namespace:
        value = namespace[MAP_ATTRIBUTE]
    else:
        value = next((vars(cls)[MAP_ATTRIBUTE] for cls in type(model).__mro__
                      if MAP_ATTRIBUTE in vars(cls)), {})
    if type(value) is not dict or len(value) > MAX_STATES:
        raise StateContractError("kernelbench_state_map must be a plain data dictionary")

    def plain(item, depth=0):
        if depth > 8:
            raise StateContractError("State mapping is too deeply nested")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise StateContractError("State mapping keys must be strings")
            return {key: plain(val, depth + 1) for key, val in item.items()}
        if type(item) is list:
            return [plain(val, depth + 1) for val in item]
        if type(item) in (str, int) or item is None:
            return item
        raise StateContractError("State mapping accepts JSON strings, integers, lists and dictionaries only")

    return plain(value)


def _storage(tensor):
    return str(tensor.device), tensor.untyped_storage()._cdata


def _span(tensor):
    if tensor.numel() == 0:
        return None
    if any(stride < 0 for stride in tensor.stride()):
        raise StateContractError("Negative state strides are unsupported")
    start = tensor.storage_offset() * tensor.element_size()
    stop = (tensor.storage_offset() + sum((size - 1) * stride for size, stride in
                                         zip(tensor.shape, tensor.stride())) + 1) * tensor.element_size()
    return _storage(tensor), start, stop


def _view(target, spec):
    if type(spec) is str:
        spec = {"target": spec}
    if (type(spec) is not dict or "target" not in spec
            or set(spec) - {"target", "slice", "permute", "reshape"}
            or type(spec["target"]) is not str or not spec["target"]):
        raise StateContractError("Invalid declarative state view")
    value = target.detach()
    if "slice" in spec:
        slices = spec["slice"]
        if type(slices) is not list or len(slices) != value.ndim:
            raise StateContractError("State slice must specify each target dimension")
        for dim, bounds in enumerate(slices):
            if bounds is None:
                continue
            if (type(bounds) is not list or len(bounds) != 2
                    or any(type(index) is not int for index in bounds)
                    or not 0 <= bounds[0] < bounds[1] <= value.shape[dim]):
                raise StateContractError("State slices need bounded nonempty [start, stop] ranges")
            value = value.narrow(dim, bounds[0], bounds[1] - bounds[0])
    if "permute" in spec:
        order = spec["permute"]
        if (type(order) is not list or any(type(dim) is not int for dim in order)
                or sorted(order) != list(range(value.ndim))):
            raise StateContractError("State permutation must be a dimension bijection")
        value = value.permute(order)
    if "reshape" in spec:
        shape = spec["reshape"]
        if type(shape) is not list or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise StateContractError("State reshape needs explicit positive dimensions")
        # view() rejects copies. An allocated reshaped tensor would not update the model.
        try:
            value = value.view(shape)
        except RuntimeError as exc:
            raise StateContractError("State reshape must remain a storage view") from exc
    if _storage(value) != _storage(target) or torch._debug_has_internal_overlap(value) != 0:
        raise StateContractError("State views must not copy or overlap themselves")
    return value


@dataclass
class Binding:
    name: str
    kind: str
    reference: torch.Tensor
    target_name: str
    target: torch.Tensor
    value: torch.Tensor
    reference_span: tuple | None
    target_span: tuple | None
    metadata: tuple
    actual_kind: str


class Bindings:
    def __init__(self, reference, candidate, *, policy=None, input_device=None):
        self.reference, self.candidate = reference, candidate
        self.policy, self.input_device = policy, input_device
        self.mapping = _declaration(candidate)
        ref_states, _ = _registered(reference)
        cand_states, _ = _registered(candidate)
        ordinary = plain_tensors(reference)
        rules = policy["ordinary"] if policy is not None else {
            name: {"role": "persistent", "replacement": "reference_matched", "initial_placement": "reference"}
            for name in ordinary}
        if ordinary.keys() != rules.keys():
            raise StateContractError("Reference ordinary state differs from its task policy")
        self.ignored = [{"reference": name, **rule} for name, rule in rules.items() if rule["role"] == "unused_initialization"]
        self.required_plain = {name for name, rule in rules.items() if rule["role"] == "persistent"}
        self.plain_rules = rules
        ref_states.update({name: ("tensor", ordinary[name]) for name in sorted(self.required_plain)})
        cand_states.update({name: ("tensor", tensor) for name, tensor in plain_tensors(candidate).items()})
        if set(self.mapping) - ref_states.keys():
            raise StateContractError("State mapping contains unknown reference names")
        self.rows = []
        for name, (kind, expected) in ref_states.items():
            spec = self.mapping.get(name, name)
            target_name = spec if type(spec) is str else spec.get("target") if type(spec) is dict else None
            if target_name not in cand_states:
                raise StateContractError(f"Required {kind} {name} has no registered candidate target or ordinary tensor target")
            actual_kind, target = cand_states[target_name]
            if kind != actual_kind and not (kind == "tensor" and actual_kind == "buffer"):
                raise StateContractError(f"State kind differs for {name}")
            actual = _view(target, spec)
            placement = rules.get(name, {}).get("initial_placement") == "cpu_or_input_device"
            allowed_device = (placement and expected.device.type == "cpu" and input_device is not None
                              and target.device == torch.device(input_device))
            if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.device != actual.device and not allowed_device:
                raise StateContractError(f"State shape, dtype or device differs for {name}")
            if expected.requires_grad != target.requires_grad:
                raise StateContractError(f"State requires_grad differs for {name}")
            self.rows.append(Binding(name, kind, expected, target_name, target, actual,
                                     _span(expected), _span(target),
                                     (expected.shape, expected.dtype, expected.device, expected.requires_grad), actual_kind))
        self.reference_names = set(ref_states)
        self._aliases()
        alias_groups = {identity: index for index, identity in enumerate(dict.fromkeys(id(row.reference) for row in self.rows))}
        self.description = [{"reference": row.name, "kind": row.kind, "target": row.target_name,
                             "view": self.mapping.get(row.name, row.name), "shape": list(row.reference.shape),
                             "dtype": str(row.reference.dtype), "alias_group": alias_groups[id(row.reference)],
                             "reference_owner": _owner(reference, row.name)[0],
                             "target_owner": _owner(candidate, row.target_name)[0]}
                            for row in self.rows]
        for row in self.description:
            if row["kind"] == "tensor":
                row["replacement"] = "reference_matched"
                row["target_kind"] = next(item.actual_kind for item in self.rows if item.name == row["reference"])
        self.sha256 = hashlib.sha256(json.dumps(self.description, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.mode_pairs = [("", "")]
        for row in self.rows:
            pair = (_owner(reference, row.name)[0], _owner(candidate, row.target_name)[0])
            if pair not in self.mode_pairs:
                self.mode_pairs.append(pair)

    def modes(self):
        records = []
        for ref_name, cand_name in self.mode_pairs:
            expected = _module(self.reference, ref_name).training
            actual = _module(self.candidate, cand_name).training
            if type(expected) is not bool or type(actual) is not bool:
                raise StateContractError("Required module training flags must be booleans")
            records.append({"reference": ref_name, "target": cand_name, "expected": expected, "actual": actual})
        return records

    def _aliases(self):
        # Exact tied tensors are supported. Different source tensors cannot map to
        # overlapping target spans; ambiguous interleaved views fail conservatively.
        by_storage, tied = {}, {}
        for row in self.rows:
            span = _span(row.value)
            key = id(row.reference)
            signature = (span, tuple(row.value.shape), row.value.stride())
            if key in tied:
                if tied[key] != signature:
                    raise StateContractError("A tied reference tensor must retain one candidate storage view")
                continue
            tied[key] = signature
            if span is not None:
                by_storage.setdefault(span[0], []).append(span[1:])
        for spans in by_storage.values():
            ordered = sorted(spans)
            if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
                raise StateContractError("Independent reference states overlap in the candidate mapping")

    def issues(self, tolerance):
        issues = []
        if _declaration(self.candidate) != self.mapping:
            return ["Candidate changed its state mapping during the audit"]
        ref_states, _ = _registered(self.reference)
        cand_states, _ = _registered(self.candidate)
        plain = plain_tensors(self.reference)
        if plain.keys() != self.plain_rules.keys():
            return ["Reference ordinary state schema changed; task contract is incomplete"]
        ref_states.update({name: ("tensor", plain[name]) for name in sorted(self.required_plain)})
        cand_states.update({name: ("tensor", tensor) for name, tensor in plain_tensors(self.candidate).items()})
        if set(ref_states) != self.reference_names:
            return ["Reference registered state schema changed; contract is incomplete"]
        for row in self.rows:
            source = ref_states[row.name][1]
            pair = cand_states.get(row.target_name)
            if row.kind == "tensor":
                if pair is None or pair[0] != row.actual_kind:
                    issues.append(f"Required ordinary state {row.name} was removed or changed kind")
                    continue
                row.reference, row.target = source, pair[1]
                row.value = _view(pair[1], self.mapping.get(row.name, row.name))
                value = row.value
                if (source.shape != value.shape or source.dtype != value.dtype
                        or source.requires_grad != pair[1].requires_grad):
                    issues.append(f"Required ordinary state {row.name} changed representation")
                    continue
                placement = self.plain_rules[row.name]["initial_placement"] == "cpu_or_input_device"
                if source.device != value.device:
                    if not (placement and source.device.type == "cpu" and self.input_device is not None
                            and value.device == torch.device(self.input_device)):
                        issues.append(f"Required ordinary state {row.name} changed device")
                        continue
                    value = value.detach().to(device=source.device)
                if issue := compare_outputs(source.detach(), value, tolerance, f"state.{row.name}"):
                    issues.append(issue)
                continue
            if (source is not row.reference or pair is None or pair[0] != row.kind
                    or pair[1] is not row.target or _span(pair[1]) != row.target_span
                    or _span(source) != row.reference_span
                    or (source.shape, source.dtype, source.device, source.requires_grad) != row.metadata):
                issues.append(f"Required state {row.name} was replaced or removed")
                continue
            value = _view(pair[1], self.mapping.get(row.name, row.name))
            if source.device != value.device or source.requires_grad != pair[1].requires_grad:
                issues.append(f"Required state {row.name} changed device or requires_grad")
                continue
            if issue := compare_outputs(source.detach(), value, tolerance, f"state.{row.name}"):
                issues.append(issue)
        self._aliases()
        return issues

    def perturb(self, step):
        seen = set()
        count = 0
        for index, row in enumerate(self.rows):
            if row.kind != "parameter" or id(row.reference) in seen or not row.reference.numel():
                continue
            seen.add(id(row.reference))
            if not (row.reference.is_floating_point() or row.reference.is_complex()):
                raise StateContractError("Parameter perturbations require floating or complex state")
            scale = 1.125 if step == 0 else 0.875
            offset = (1 if step == 0 else -1) * (index % 7 + 1) / 32
            for tensor in (row.reference, row.value):
                tensor.mul_(scale).add_(offset)
            count += 1
        return count


def _snapshot_aliases(model):
    states, _ = _registered(model)
    values = {name: tensor for name, (_, tensor) in states.items()}
    values.update(plain_tensors(model))
    identities, objects, storages = {}, [], {}
    for name in sorted(values):
        tensor = values[name]
        index = identities.setdefault(id(tensor), len(identities))
        objects.append((name, index))
        if span := _span(tensor):
            storages.setdefault(span[0], {})[index] = (*span[1:], index)
    overlap = []
    for entries in storages.values():
        component, stop = [], -1

        def record():
            if len(component) > 1:
                base = component[0][0]
                overlap.append(sorted((index, start - base, end - base) for start, end, index in component))

        for start, end, index in sorted(entries.values()):
            if start >= stop:
                record()
                component, stop = [], -1
            component.append((start, end, index))
            stop = max(stop, end)
        record()
    return objects, sorted(overlap)


def _isolated_pair(reference, candidate):
    originals = [_registered(model) for model in (reference, candidate)]
    ref, cand = copy.deepcopy(reference), copy.deepcopy(candidate)
    snapshots = [_registered(model) for model in (ref, cand)]
    original_modules = originals[0][1] | originals[1][1]
    original_storage = {_storage(tensor) for states, _ in originals for _, tensor in states.values() if tensor.numel()}
    original_storage.update(_storage(tensor) for model in (reference, candidate)
                            for tensor in _plain_tensors(model).values() if tensor.numel())
    original_storage.update(_storage(tensor) for model in (reference, candidate)
                            for tensor in plain_tensors(model).values() if tensor.numel())
    if (snapshots[0][1] & snapshots[1][1]
            or any(modules & original_modules for _, modules in snapshots)):
        raise StateContractError("Model snapshot reused an original or peer module")
    stores = [{_storage(tensor) for _, tensor in states.values() if tensor.numel()} for states, _ in snapshots]
    for model, storage in zip((ref, cand), stores):
        storage.update(_storage(tensor) for tensor in _plain_tensors(model).values() if tensor.numel())
        storage.update(_storage(tensor) for tensor in plain_tensors(model).values() if tensor.numel())
    if stores[0] & stores[1] or any(storage & original_storage for storage in stores):
        raise StateContractError("Model snapshot reused original or peer tensor storage")
    for (original, _), (cloned, _) in zip(originals, snapshots):
        if original.keys() != cloned.keys():
            raise StateContractError("Model snapshot changed the registered state schema")
        for name, (kind, tensor) in original.items():
            copied_kind, copied_tensor = cloned[name]
            if (kind != copied_kind or tensor.device != copied_tensor.device
                    or tensor.requires_grad != copied_tensor.requires_grad
                    or compare_outputs(tensor.detach(), copied_tensor.detach(), 0)):
                raise StateContractError("Model snapshot changed registered tensor values or metadata")
    for original, cloned in zip((reference, candidate), (ref, cand)):
        if _snapshot_aliases(original) != _snapshot_aliases(cloned):
            raise StateContractError("Model snapshot changed tensor object aliases or overlapping state views")
        before, after = plain_tensors(original), plain_tensors(cloned)
        if before.keys() != after.keys():
            raise StateContractError("Model snapshot changed ordinary tensor paths")
        for name, value in before.items():
            copied = after[name]
            if ((value.device, value.dtype, value.shape, value.stride(), value.requires_grad) !=
                    (copied.device, copied.dtype, copied.shape, copied.stride(), copied.requires_grad)
                    or compare_outputs(value.detach(), copied.detach(), 0)):
                raise StateContractError("Model snapshot changed ordinary tensor values or metadata")
    return ref, cand


def _module(model, path):
    for segment in path.split(".") if path else ():
        model = vars(model)["_modules"][segment]
    return model


def _owner(model, state_name):
    if not state_name.startswith("@"):
        path = state_name.rpartition(".")[0]
        return path, _module(model, path)
    current, parts = model, []
    for segment in state_name[1:].split("."):
        if segment not in vars(current)["_modules"]:
            break
        current = vars(current)["_modules"][segment]
        parts.append(segment)
    return ".".join(parts), current


def state_checks(reference, candidate, args, *, seed, set_seed, tolerance, device, policy=None):
    """Validate state transitions and mutation on isolated copies of actual models."""
    issues, events, descriptions = [], [], []
    parameter_count = 0

    def observe(binding, label, training):
        ref, cand = binding.reference, binding.candidate
        if ref.train(training) is not ref or cand.train(training) is not cand:
            raise StateContractError("train() must update and return the audited module")
        if ref.training is not training or cand.training is not training:
            raise StateContractError("Root training mode differs from the requested mode")
        before_modes = binding.modes()
        set_seed(seed)
        expected = clone_tensors(ref(*clone_tensors(args)))
        set_seed(seed)
        actual = clone_tensors(cand(*clone_tensors(args)))
        synchronize(device)
        if ref.training is not training or cand.training is not training or torch.is_grad_enabled():
            issues.append(f"{label}: forward changed the requested mode or gradient state")
        if issue := compare_outputs(expected, actual, tolerance):
            issues.append(f"{label}: {issue}")
        issues.extend(f"{label}: {issue}" for issue in binding.issues(tolerance))
        after_modes = binding.modes()
        if before_modes != after_modes or any(row["expected"] != row["actual"] for row in after_modes):
            issues.append(f"{label}: required module modes changed or differ")
        events.append({"case": label, "training": training, "seed": seed,
                       "module_modes_before": before_modes, "module_modes_after": after_modes,
                       "no_grad_after": not torch.is_grad_enabled()})
        return expected

    try:
        if (type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0
                or type(seed) is not int):
            raise StateContractError("State audit requires a finite nonnegative tolerance and integer seed")
        with torch.no_grad():
            ref, cand = _isolated_pair(reference, candidate)
            binding = Bindings(ref, cand, policy=policy, input_device=device)
            descriptions.append({"phase": "modes", "sha256": binding.sha256, "states": binding.description,
                                 "ignored_ordinary": binding.ignored})
            parameter_count = len({id(row.reference) for row in binding.rows if row.kind == "parameter"})
            issues.extend(f"initial: {issue}" for issue in binding.issues(tolerance))
            for index, training in enumerate(MODE_SEQUENCE):
                observe(binding, f"mode={training}/call={index}", training)
            if parameter_count:
                for training in (True, False):
                    ref, cand = _isolated_pair(reference, candidate)
                    binding = Bindings(ref, cand, policy=policy, input_device=device)
                    descriptions.append({"phase": f"parameters/training={training}", "sha256": binding.sha256,
                                         "states": binding.description, "ignored_ordinary": binding.ignored})
                    previous = observe(binding, f"parameter warmup/training={training}", training)
                    for step in range(2):
                        changed = binding.perturb(step)
                        expected = observe(binding, f"after parameter update {step}/training={training}", training)
                        events[-1].update(parameters_updated=changed,
                                          reference_output_changed=compare_outputs(previous, expected, 0) is not None)
                        previous = expected
    except Exception as exc:
        issues.append(f"State audit unavailable: {type(exc).__name__}: {str(exc)[:1000]}")
    return check("failed" if issues else "passed",
                 reason="Declared tensor state, persistent train/eval transitions and same-object parameter updates",
                 values={"protocol": PROTOCOL, "issues": issues, "parameter_count": parameter_count,
                         "bindings": descriptions, "events": events,
                         "security_isolation_certified": False})

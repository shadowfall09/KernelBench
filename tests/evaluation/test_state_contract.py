"""Functional state audits on CPU only, including adversarial representations."""

import copy
import json

import pytest
import torch

from kb_workflows.evaluation.state_contract import Bindings, StateContractError, state_checks


def audit(reference, candidate, args=None):
    return state_checks(reference, candidate, args or [torch.randn(4, 3)], seed=73,
                        set_seed=torch.manual_seed, tolerance=1e-5, device=torch.device("cpu"))


def assert_failed(result, message):
    assert result["status"] == "failed", result
    assert any(message in issue for issue in result["values"]["issues"]), result


class Renamed(torch.nn.Module):
    kernelbench_state_map = {"weight": "layers.0.weight", "bias": "layers.0.bias"}

    def __init__(self, linear):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(linear)])

    def forward(self, x):
        return self.layers[0](x)


def test_nested_renaming_keeps_original_models_and_input_untouched():
    reference = torch.nn.Linear(3, 2)
    candidate = Renamed(reference)
    x = torch.randn(4, 3)
    saved = [copy.deepcopy(model.state_dict()) for model in (reference, candidate)]
    before = x.clone()
    result = audit(reference, candidate, [x])
    assert result["status"] == "passed", result
    values = result["values"]
    assert values["parameter_count"] == 2
    assert len(values["events"]) == 11
    assert [row["training"] for row in values["events"][:5]] == [True, True, False, True, False]
    assert all(row["parameters_updated"] == 2 for row in values["events"] if "parameters_updated" in row)
    assert all(len(row["sha256"]) == 64 for row in values["bindings"])
    for model, snapshot in zip((reference, candidate), saved):
        assert all(torch.equal(tensor, snapshot[name]) for name, tensor in model.state_dict().items())
    assert torch.equal(x, before)
    json.dumps(result, allow_nan=False)


class Dual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(3, 2)
        self.b = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.a(x), self.b(x)


class Packed(torch.nn.Module):
    kernelbench_state_map = {
        "a.weight": {"target": "weights", "slice": [[0, 2], None]},
        "b.weight": {"target": "weights", "slice": [[2, 4], None]},
        "a.bias": {"target": "biases", "slice": [[0, 2]]},
        "b.bias": {"target": "biases", "slice": [[2, 4]]},
    }

    def __init__(self, reference):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.cat([reference.a.weight, reference.b.weight]).detach())
        self.biases = torch.nn.Parameter(torch.cat([reference.a.bias, reference.b.bias]).detach())

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weights, self.biases)
        return out[:, :2], out[:, 2:]


def test_fused_parameter_slices_update_the_actual_backing_parameter():
    reference = Dual()
    candidate = Packed(reference)
    result = audit(reference, candidate)
    assert result["status"] == "passed", result
    assert result["values"]["parameter_count"] == 4
    with torch.no_grad():
        binding = Bindings(reference, candidate)
        binding.perturb(0)
        assert torch.equal(reference.a.weight, candidate.weights[:2])
        assert torch.equal(reference.b.weight, candidate.weights[2:])


@pytest.mark.parametrize("representation", ["transpose", "flat"])
def test_transposed_and_flattened_registered_weights(representation):
    reference = torch.nn.Linear(3, 2, bias=False)

    class Candidate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if representation == "transpose":
                self.kernelbench_state_map = {"weight": {"target": "w", "permute": [1, 0]}}
                self.w = torch.nn.Parameter(reference.weight.detach().t().contiguous())
            else:
                self.kernelbench_state_map = {"weight": {"target": "w", "reshape": [2, 3]}}
                self.w = torch.nn.Parameter(reference.weight.detach().flatten().clone())

        def forward(self, x):
            return x @ self.w if representation == "transpose" else x @ self.w.view(2, 3).t()

    result = audit(reference, Candidate())
    assert result["status"] == "passed", result


def test_batch_norm_buffers_can_move_to_a_different_nested_module_name():
    reference = torch.nn.BatchNorm1d(3)

    class Candidate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.ModuleDict({"norm": copy.deepcopy(reference)})
            self.kernelbench_state_map = {name: "block.norm." + name for name in reference.state_dict()}

        def forward(self, x):
            return self.block["norm"](x)

    result = audit(reference, Candidate())
    assert result["status"] == "passed", result
    names = {row["reference"] for row in result["values"]["bindings"][0]["states"]}
    assert {"running_mean", "running_var", "num_batches_tracked"} <= names


def test_zero_parameter_cache_fails_nonzero_update_in_both_modes():
    reference = torch.nn.Linear(3, 2)
    torch.nn.init.zeros_(reference.weight)
    torch.nn.init.zeros_(reference.bias)

    class Cached(torch.nn.Linear):
        def forward(self, x):
            if not hasattr(self, "saved_weight"):
                self.saved_weight, self.saved_bias = self.weight.detach().clone(), self.bias.detach().clone()
            return torch.nn.functional.linear(x, self.saved_weight, self.saved_bias)

    candidate = Cached(3, 2)
    candidate.load_state_dict(reference.state_dict())
    result = audit(reference, candidate)
    assert_failed(result, "after parameter update")
    issues = result["values"]["issues"]
    assert any("training=True" in issue for issue in issues)
    assert any("training=False" in issue for issue in issues)


def test_persistent_sequence_catches_second_training_call_state_bug():
    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros((), dtype=torch.int64), persistent=False)

        def forward(self, x):
            if self.training:
                self.count.add_(1)
            return x

    class Broken(Counter):
        def forward(self, x):
            if self.training and self.count.item() == 0:
                self.count.add_(1)
            return x

    result = audit(Counter(), Broken())
    assert_failed(result, "mode=True/call=1: state.count")


def test_initial_state_mismatch_fails_even_when_forward_overwrites_it():
    class Reset(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("buf", torch.tensor(value))

        def forward(self, x):
            self.buf.zero_()
            return x

    assert_failed(audit(Reset(1.0), Reset(7.0)), "initial: state.buf")


class Tied(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 3))
        self.alias = self.weight

    def forward(self, x):
        return x @ self.weight + x @ self.alias


def test_tied_parameters_are_updated_once_and_must_remain_tied():
    reference = Tied()
    result = audit(reference, copy.deepcopy(reference))
    assert result["status"] == "passed", result
    assert result["values"]["parameter_count"] == 1
    candidate = copy.deepcopy(reference)
    candidate.alias = torch.nn.Parameter(candidate.weight.detach().clone())
    assert_failed(audit(reference, candidate), "tied reference tensor")


def test_independent_parameters_cannot_alias_even_when_values_initially_match():
    reference = Tied()
    reference.alias = torch.nn.Parameter(reference.weight.detach().clone())
    candidate = Tied()
    candidate.weight.data.copy_(reference.weight)
    assert_failed(audit(reference, candidate), "Independent reference states overlap")


@pytest.mark.parametrize("mapping, expected", [
    ({"unknown": "weight"}, "unknown reference names"),
    ({"weight": "missing"}, "no registered candidate target"),
    ({"weight": {"target": "weight", "function": "identity"}}, "Invalid declarative"),
    ({"weight": {"target": "weight", "slice": [[-1, 2], None]}}, "bounded nonempty"),
    ({"weight": {"target": "weight", "slice": [[0, 2]]}}, "each target dimension"),
    ({"weight": {"target": "weight", "slice": [[False, 2], None]}}, "JSON strings, integers"),
    ({"weight": {"target": "weight", "permute": [0, 0]}}, "dimension bijection"),
    ({"weight": {"target": "weight", "reshape": [-1, 3]}}, "positive dimensions"),
    ({"weight": {"target": "weight", "reshape": [6]}}, "shape, dtype or device"),
    ({"weight": 7}, "no registered candidate target"),
])
def test_invalid_state_views_never_qualify(mapping, expected):
    reference = torch.nn.Linear(3, 2)
    candidate = copy.deepcopy(reference)
    candidate.kernelbench_state_map = mapping
    assert_failed(audit(reference, candidate), expected)


@pytest.mark.parametrize("kind", ["method", "property", "dict_subclass", "tensor"])
def test_mapping_is_plain_data_and_does_not_execute_candidate_mapping_code(kind):
    calls = []

    class Candidate(torch.nn.Linear):
        pass

    if kind in {"method", "property"}:
        def declaration(self):
            calls.append(True)
            return {}
        Candidate.kernelbench_state_map = property(declaration) if kind == "property" else declaration
    else:
        class Fancy(dict):
            pass
        Candidate.kernelbench_state_map = Fancy() if kind == "dict_subclass" else {"weight": torch.tensor(1)}
    reference = torch.nn.Linear(3, 2)
    candidate = Candidate(3, 2)
    candidate.load_state_dict(reference.state_dict())
    assert_failed(audit(reference, candidate), "State audit unavailable")
    assert calls == []


@pytest.mark.parametrize("kind", ["dtype", "shape", "kind", "requires_grad", "unregistered"])
def test_state_schema_mismatches_cannot_be_excused_by_correct_outputs(kind):
    reference = torch.nn.Linear(3, 2)
    candidate = Renamed(reference)
    layer = candidate.layers[0]
    if kind == "dtype":
        layer.weight.data = layer.weight.data.double()
    elif kind == "shape":
        layer.weight.data = layer.weight.data.t().contiguous()
    elif kind == "kind":
        value = layer.weight.detach()
        del layer.weight
        layer.register_buffer("weight", value)
    elif kind == "requires_grad":
        layer.weight.requires_grad_(False)
    else:
        value = layer.weight.detach()
        del layer.weight
        layer.weight = value
    assert_failed(audit(reference, candidate), "State audit unavailable")


@pytest.mark.parametrize("shared", ["module", "storage"])
def test_bad_model_snapshots_fail_without_mutating_original_state(shared):
    class Candidate(torch.nn.Linear):
        def __deepcopy__(self, memo):
            if shared == "module":
                return self
            result = torch.nn.Linear(3, 2)
            result.weight.data = self.weight.data
            result.bias.data = self.bias.data
            return result

    reference = torch.nn.Linear(3, 2)
    candidate = Candidate(3, 2)
    candidate.load_state_dict(reference.state_dict())
    saved = candidate.weight.detach().clone()
    assert_failed(audit(reference, candidate), "snapshot reused")
    assert torch.equal(candidate.weight, saved)


def test_map_mutation_and_parameter_replacement_are_detected():
    class Mutating(Renamed):
        def forward(self, x):
            self.kernelbench_state_map = {}
            return super().forward(x)

    reference = torch.nn.Linear(3, 2)
    assert_failed(audit(reference, Mutating(reference)), "changed its state mapping")

    class Replacing(torch.nn.Linear):
        def forward(self, x):
            self.weight = torch.nn.Parameter(self.weight.detach().clone())
            return super().forward(x)

    candidate = Replacing(3, 2)
    candidate.load_state_dict(reference.state_dict())
    assert_failed(audit(reference, candidate), "was replaced or removed")


def test_requested_mode_and_gradient_state_are_checked_and_grad_is_restored():
    class BadMode(torch.nn.Identity):
        def forward(self, x):
            self.training = False
            torch.set_grad_enabled(True)
            return x

    before = torch.is_grad_enabled()
    assert_failed(audit(torch.nn.Identity(), BadMode()), "changed the requested mode or gradient state")
    assert torch.is_grad_enabled() == before


def test_dropout_and_complex_parameters_replay_rng_and_state_on_cpu():
    reference = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.4))
    result = audit(reference, copy.deepcopy(reference))
    assert result["status"] == "passed", result
    reference = torch.nn.Linear(3, 2, dtype=torch.complex128)
    result = audit(reference, copy.deepcopy(reference), [torch.randn(4, 3, dtype=torch.complex128)])
    assert result["status"] == "passed", result


def test_reshape_that_would_copy_storage_is_rejected():
    reference = torch.nn.Linear(3, 2, bias=False)
    candidate = copy.deepcopy(reference)
    candidate.kernelbench_state_map = {"weight": {"target": "weight", "permute": [1, 0], "reshape": [2, 3]}}
    assert_failed(audit(reference, candidate), "remain a storage view")


def test_overlapping_packed_slices_are_rejected_before_calls():
    reference = Dual()
    candidate = Packed(reference)
    candidate.kernelbench_state_map = copy.deepcopy(Packed.kernelbench_state_map)
    candidate.kernelbench_state_map["b.weight"]["slice"][0] = [1, 3]
    with pytest.raises(StateContractError, match="overlap"):
        Bindings(reference, candidate)


def test_reference_execution_error_is_explicit_failure():
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("reference input is illegal")

    assert_failed(audit(Broken(), torch.nn.Identity()), "reference input is illegal")


@pytest.mark.parametrize("kind", ["data_storage", "requires_grad"])
def test_forward_cannot_silently_replace_parameter_storage_or_gradient_metadata(kind):
    class Candidate(torch.nn.Linear):
        def forward(self, x):
            if kind == "data_storage":
                self.weight.data = self.weight.detach().clone()
            else:
                self.weight.requires_grad_(False)
            return super().forward(x)

    reference = torch.nn.Linear(3, 2)
    candidate = Candidate(3, 2)
    candidate.load_state_dict(reference.state_dict())
    assert_failed(audit(reference, candidate), "Required state weight")


def test_snapshot_cannot_change_values_to_hide_bad_original_state():
    class Candidate(torch.nn.Linear):
        def __deepcopy__(self, memo):
            result = torch.nn.Linear(3, 2)
            result.load_state_dict(self.state_dict())
            result.weight.data.zero_()
            return result

    reference = torch.nn.Linear(3, 2)
    candidate = Candidate(3, 2)
    candidate.load_state_dict(reference.state_dict())
    assert_failed(audit(reference, candidate), "snapshot changed registered tensor values")


def test_unregistered_reference_tensor_state_cannot_silently_escape_the_contract():
    class Plain(torch.nn.Identity):
        def __init__(self):
            super().__init__()
            self.hidden = {"states": [torch.ones(3)]}

    reference = Plain()
    assert audit(reference, copy.deepcopy(reference))["status"] == "passed"
    candidate = copy.deepcopy(reference)
    candidate.hidden["states"][0].zero_()
    assert_failed(audit(reference, candidate), 'state.@hidden["states"][0]')


@pytest.mark.parametrize("level,pid,init,args", [
    (2, 33, [3, 4, (4,)], lambda: [torch.randn(5, 3)]),
    (3, 2, [3, [5, 4], 2], lambda: [torch.randn(5, 3)]),
    (3, 34, [3, 4, 2], lambda: [torch.randn(3, 5, 3), torch.randn(5, 4)]),
])
def test_actual_reference_stateful_and_nested_modules_on_explicit_small_cpu_inputs(level, pid, init, args):
    from kb_workflows.common import problem_paths
    from src.eval import load_original_model_and_inputs

    Model, _, _ = load_original_model_and_inputs(problem_paths(level)[pid].read_text(), {})
    reference = Model(*init)
    result = audit(reference, copy.deepcopy(reference), args())
    assert result["status"] == "passed", result

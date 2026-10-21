"""CPU regressions for the standard and hidden output correctness contracts."""

import pytest
import torch
from torch import nn

from src.eval import (
    _process_input_tensor,
    run_and_check_correctness,
    run_correctness_on_fixed_inputs,
)
from src.validation import clone_tensors, compare_outputs


class FunctionModel(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


@pytest.fixture(params=["standard", "hidden"])
def check(request):
    def evaluate(reference, candidate, inputs, precision=torch.float32):
        metadata = {}
        if request.param == "standard":
            result = run_and_check_correctness(
                reference, candidate, lambda: inputs, metadata,
                num_correct_trials=2, device="cpu", precision=precision,
            )
            return result.correctness, result.metadata
        passed = run_correctness_on_fixed_inputs(
            reference, candidate, inputs, metadata, device="cpu", precision=precision,
        )
        return passed, metadata

    return evaluate


@pytest.mark.parametrize(
    "candidate, issue",
    [
        (lambda x: {"indices": x + 1, "values": x.float()}, "discrete"),
        (lambda x: {"indices": x.float(), "values": x.float()}, "dtype"),
        (lambda x: {"indices": x.reshape(1, -1), "values": x.float()}, "shape"),
        (lambda x: {"wrong_key": x, "values": x.float()}, "dictionary"),
        (lambda x: [x, x.float()], "dictionary"),
    ],
)
def test_nested_output_mismatches_reject_candidate(check, candidate, issue):
    reference = FunctionModel(lambda x: {"indices": x, "values": x.float()})
    # An off-by-one index at this scale passed a relative floating tolerance.
    passed, metadata = check(reference, FunctionModel(candidate), [torch.tensor([100000, 200000])])
    assert not passed
    assert issue in str(metadata)


def test_matching_nested_outputs_pass(check):
    def forward(x):
        return {"values": [x, x.square()], "indices": (torch.tensor([100000]), x > 0)}

    passed, _ = check(FunctionModel(forward), FunctionModel(forward), [torch.tensor([-1.0, 2.0])])
    assert passed


def test_bool_outputs_are_exact_even_with_large_tolerance():
    assert compare_outputs(torch.tensor([True]), torch.tensor([False]), tolerance=1)


def test_list_and_tuple_are_distinct():
    assert compare_outputs([torch.ones(2)], (torch.ones(2),), tolerance=1e-3)
    assert compare_outputs([torch.ones(2)], [torch.ones(2), torch.ones(2)], tolerance=1e-3)


@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_discrete_inputs_preserve_dtype(precision, dtype):
    values = torch.tensor([0, 1], dtype=dtype)
    converted = _process_input_tensor(values, "cpu", precision)
    assert converted.dtype == dtype
    assert torch.equal(converted, values)


@pytest.mark.parametrize("replacement", [None, (0, 0.0), (1, -float("inf")), (2, float("inf")), (3, 2.0)])
def test_nonfinite_masks_signs_and_finite_values(check, replacement):
    values = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.5])

    def candidate(x):
        output = x.clone()
        if replacement is not None:
            output[replacement[0]] = replacement[1]
        return output

    passed, _ = check(nn.Identity(), FunctionModel(candidate), [values])
    assert passed is (replacement is None)


@pytest.mark.parametrize("offset, accepted", [(0.0005, True), (0.02, False)])
def test_finite_float_tolerance_is_preserved(check, monkeypatch, offset, accepted):
    monkeypatch.delenv("KB_FP32_TOL", raising=False)
    passed, _ = check(nn.Identity(), FunctionModel(lambda x: x + offset), [torch.ones(4)])
    assert passed is accepted


def test_matching_inplace_models_receive_independent_inputs(check):
    inputs = [torch.ones(4)]
    passed, _ = check(FunctionModel(lambda x: x.add_(1)), FunctionModel(lambda x: x.add_(1)), inputs)
    assert passed
    assert torch.equal(inputs[0], torch.ones(4))


def test_candidate_mutation_cannot_overwrite_reference_answer(check):
    inputs = [torch.ones(4)]
    passed, _ = check(nn.Identity(), FunctionModel(lambda x: x.add_(1)), inputs)
    assert not passed
    assert torch.equal(inputs[0], torch.ones(4))


def test_reference_output_is_snapshotted_before_candidate_reuses_buffer(check):
    buffer = torch.empty(4)

    def reference(x):
        return buffer.copy_(x)

    def candidate(x):
        return buffer.copy_(x + 1)

    passed, _ = check(FunctionModel(reference), FunctionModel(candidate), [torch.ones(4)])
    assert not passed


def test_nested_input_copies_preserve_values_and_noncontiguous_strides():
    view = torch.arange(24.0).reshape(4, 6)[:, ::2]
    inputs = {"args": (view, [torch.tensor([True, False])])}
    copied = clone_tensors(inputs)
    assert copied["args"][0].stride() == view.stride()
    copied["args"][0].add_(100)
    copied["args"][1][0].logical_not_()
    assert torch.equal(view, torch.arange(24.0).reshape(4, 6)[:, ::2])
    assert torch.equal(inputs["args"][1][0], torch.tensor([True, False]))


def test_matching_randomized_models_use_the_same_seed(check):
    passed, _ = check(nn.Dropout(0.5), nn.Dropout(0.5), [torch.ones(128)])
    assert passed

"""Independent formulas versus frozen L1 sources on small explicit CPU inputs."""

import copy
import json

import pytest
import torch

from kb_workflows import operator_oracles
from kb_workflows.common import problem_paths
from kb_workflows.numerical import PrecisionContract, precision_gate
from kb_workflows.operator_specs import spec_for
from src.eval import load_original_model_and_inputs

POLICY = PrecisionContract(5e-6, 1e-7, 5e-5, 5e-7, 1e-6)


@pytest.fixture(autouse=True)
def cpu_only_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_case(pid, dtype=torch.float32):
    torch.manual_seed(10000 + pid)
    source = problem_paths(1)[pid].read_text()
    Model, _, _ = load_original_model_and_inputs(source, {})
    init, args = [], None
    def rand(*shape):
        return torch.randn(shape, dtype=dtype)
    if pid in {1, 13, 14, 15}:
        args = [rand(7, 7), rand(7, 7)]
        if pid == 13:
            args = [(value + value.t()) / 2 for value in args]
        if pid in {14, 15}:
            args = [value.triu() if pid == 14 else value.tril() for value in args]
    elif pid in {2, 6, 7, 8, 9}:
        args = [rand(5, 7), rand(7, 3)]
    elif pid == 3:
        args = [rand(2, 5, 7), rand(2, 7, 3)]
    elif pid == 4:
        args = [rand(5, 7), rand(7, 1)]
    elif pid == 5:
        args = [rand(5, 7), 1.75]
    elif pid in {10, 11}:
        args = [rand(*((2, 5, 7) if pid == 10 else (2, 3, 5, 7))), rand(7, 3)]
    elif pid == 12:
        args = [rand(7), rand(7, 3)]
    elif pid == 16:
        args = [rand(7, 5), rand(7, 3)]
    elif pid == 17:
        args = [rand(5, 7), rand(3, 7)]
    elif pid == 18:
        args = [rand(7, 5), rand(3, 7)]
    elif 19 <= pid <= 32 or pid == 88:
        init = [0.1] if pid == 20 else [1] if pid == 24 else [1.3] if pid == 31 else []
        args = [rand(3, 17)]
    elif pid in {33, 34, 35, 36, 37, 38, 39, 40}:
        init = [4] if pid in {33, 34, 36} else [4, 2] if pid == 35 else [(4, 3, 5)] if pid == 40 else []
        # GroupNorm takes num_features then num_groups in the frozen source.
        args = [rand(2, 4, 3, 5)] if pid in {33, 34, 35, 36, 40} else [rand(3, 7)]
    elif 41 <= pid <= 46:
        rank = (pid - 41) % 3 + 1
        init = [3, 2, 1, 1, True] if pid in {41, 43} else [3, 2, 1, 1] if pid == 42 else [3, 2, 1]
        if pid == 43:
            init.append(True)
        args = [rand(2, 3, *([7] * rank))]
    elif pid in {47, 48, 49, 51, 52, 53} or 89 <= pid <= 93:
        init = [1]
        args = [rand(3, 7)]
        if pid == 90:
            args[0] = (args[0] * 0.1 + 1).to(dtype)
        if pid == 93:
            args.append(torch.rand(3, 7) > 0.4)
    elif pid == 50:
        args = [rand(2, 3, 17, 17)]
    elif 54 <= pid <= 87:
        import inspect
        names = list(inspect.signature(Model.__init__).parameters)[1:]
        three_d = pid in {54, 58, 59, 60, 61, 66, 68, 70, 72, 73, 77}
        rank = 3 if three_d else 1 if pid in {64, 67, 74, 76, 79} else 2
        tuple_kernel = pid in {56, 58, 60, 62, 65, 66, 68, 69, 72, 75, 78, 80}
        values = {"in_channels": 4, "out_channels": 4, "kernel_size": (3,) * rank if tuple_kernel else 3,
                  "kernel_size_h": 3, "kernel_size_w": 2, "stride": 2, "stride_h": 2, "stride_w": 1,
                  "padding": 1, "padding_h": 1, "padding_w": 0, "dilation": 2, "dilation_h": 1,
                  "dilation_w": 2, "output_padding": 1, "groups": 2, "bias": True}
        init = [values[name] for name in names]
        args = [rand(2, 4, *([9] * rank))]
    elif pid in {94, 96, 98, 100}:
        args = [rand(3, 7), rand(3, 7)]
        if pid == 98:
            args = [value.softmax(-1) for value in args]
        if pid == 100:
            args[1] = args[1].sign()
    elif pid == 95:
        args = [rand(3, 7), torch.tensor([0, 3, 6], dtype=torch.int64)]
    elif pid == 97:
        args = [rand(2, 3, 5, 7), rand(2, 3, 9, 7), rand(2, 3, 9, 4)]
    elif pid == 99:
        anchor = rand(3, 7)
        init, args = [0.7], [anchor, anchor + 2, anchor.clone()]
    assert args is not None
    model = Model(*init).to(dtype=dtype)
    return source, model, args


@pytest.mark.parametrize("pid", range(1, 101))
def test_all_original_l1_operators_have_frozen_independent_cpu_formulas(pid):
    source, model, args = small_case(pid)
    spec = spec_for(1, pid, source)
    assert spec["oracle"] == "independent_cpu_fp64_formula"
    with torch.no_grad():
        truth = operator_oracles.execute(pid, model, args)
        expected = model(*copy.deepcopy(args))
        result = precision_gate(expected, expected.clone() if isinstance(expected, torch.Tensor) else copy.deepcopy(expected), truth, POLICY)
    assert result["passed"], (pid, result)
    assert result["reference_qualified"]
    assert operator_oracles.estimate_bytes(pid, model, args) > 0


@pytest.mark.parametrize("pid", [2, 11, 26, 33, 35, 40, 41, 43, 45, 54, 57, 64, 68, 75, 82, 86, 95, 97, 99])
def test_oracle_does_not_call_reference_forward_or_native_fused_operator(pid, monkeypatch):
    _, model, args = small_case(pid, torch.float64)
    with torch.no_grad():
        expected = copy.deepcopy(model)(*copy.deepcopy(args))

    def forbidden(*a, **kw):
        raise AssertionError("Independent oracle must not invoke this native reference path")

    monkeypatch.setattr(model, "forward", forbidden)
    for name in ("gelu", "batch_norm", "group_norm", "layer_norm", "conv1d", "conv2d", "conv3d",
                 "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "max_pool1d", "max_pool2d",
                 "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d", "scaled_dot_product_attention",
                 "cross_entropy", "triplet_margin_loss"):
        monkeypatch.setattr(torch.nn.functional, name, forbidden)
    with torch.no_grad():
        result = precision_gate(expected, copy.deepcopy(expected), operator_oracles.execute(pid, model, args),
                                PrecisionContract(1e-10, 1e-11, 1e-9, 1e-11, 1e-11))
    assert result["passed"], result


@pytest.mark.parametrize("pid", [20, 24, 26, 33, 34, 35, 40, 43, 55, 69, 86, 95, 97, 98, 99])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_storage_is_promoted_without_changing_role_dtypes(pid, dtype):
    source, model, args = small_case(pid, dtype)
    with torch.no_grad():
        expected = model(*copy.deepcopy(args))
        truth = operator_oracles.execute(pid, model, args)
        policy = PrecisionContract(**spec_for(1, pid, source)["precision"][str(dtype).removeprefix("torch.")])
        result = precision_gate(expected, copy.deepcopy(expected), truth, policy)
    assert result["passed"], result
    if pid == 95:
        assert args[1].dtype == torch.int64


def test_reference_qualification_cannot_hide_shape_structure_dtype_or_nonfinite_failure():
    ref = torch.ones(2)
    for truth in (torch.ones(3).double(), [torch.ones(2).double()], torch.ones(2), torch.full((2,), float("inf"), dtype=torch.float64)):
        result = precision_gate(ref, ref, truth, POLICY)
        assert not result["reference_qualified"], result
    wrong_candidate = precision_gate(ref, torch.ones(3), ref.double(), POLICY)
    assert not wrong_candidate["passed"] and wrong_candidate["reference_qualified"]
    result = precision_gate(torch.tensor([1]), torch.tensor([1]), torch.tensor([1.0]), POLICY)
    assert not result["reference_qualified"]


def test_oracle_manifest_rejects_source_drift_and_unknown_tasks():
    source, _, _ = small_case(2)
    with pytest.raises(ValueError, match="reference changed"):
        spec_for(1, 2, source + "# change")
    with pytest.raises(ValueError, match="reference changed"):
        spec_for(3, 2, source)
    with pytest.raises(ValueError, match="No independent"):
        spec_for(5, 2, source)
    json.dumps(spec_for(1, 2, source), allow_nan=False)


def test_bfloat16_reference_cancellation_is_rejected_without_relaxing_the_policy():
    source = problem_paths(1)[99].read_text()
    Model, _, _ = load_original_model_and_inputs(source, {})
    model = Model(0.7).bfloat16()
    args = [torch.zeros(1, 3, dtype=torch.bfloat16), torch.tensor([[-1., 0, 0]], dtype=torch.bfloat16),
            torch.tensor([[-1.6953125, 0, 0]], dtype=torch.bfloat16)]
    native = model(*args)
    truth = operator_oracles.execute(99, model, args)
    policy = PrecisionContract(**spec_for(1, 99, source)["precision"]["bfloat16"])
    report = precision_gate(native, native.clone(), truth, policy)
    assert not report["passed"] and not report["reference_qualified"]
    assert report["reference_issues"]


@pytest.mark.parametrize("pid", [41, 43, 51, 52])
def test_discrete_indices_match_first_tie_and_padded_pooling_semantics(pid):
    _, model, args = small_case(pid)
    args[0].fill_(1)
    expected = model(*args)
    truth = operator_oracles.execute(pid, model, args)
    assert precision_gate(expected, copy.deepcopy(expected), truth, POLICY)["passed"]


@pytest.mark.parametrize("pid", [33, 34, 35, 40])
def test_normalization_oracles_use_the_observed_eval_state_and_affine_weights(pid):
    _, model, args = small_case(pid)
    with torch.no_grad():
        model(*args)
        model.eval()
        for parameter in model.parameters():
            parameter.add_(0.2)
        expected = model(*args)
        assert precision_gate(expected, expected.clone(), operator_oracles.execute(pid, model, args), POLICY)["passed"]

"""Actual input views and numerical patterns on CPU/meta only; no experiments."""

import copy

import pytest
import torch

from kb_workflows.common import problem_paths
from kb_workflows.evaluation.measurement import input_signature
from kb_workflows.numerics import input_layouts, model_input_cases, model_stress, operator_cases
from src import eval as native
from src.input_groups import copy_inputs, layout
from src.reliability import check_paired_calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_group_transfer_and_clone_keep_offset_stride_shared_storage_and_identity(dtype):
    base = torch.arange(80.).reshape(10, 8)
    left, right = base[2:6, ::2], base[3:7, ::2]
    args = [left, right, left, torch.tensor([1, 2], dtype=torch.int64), True]
    transferred = native._process_input_group(args, "cpu", dtype)
    expected = layout(args)
    for item in expected["tensors"][:3]:
        item["dtype"] = str(dtype)
    assert layout(transferred) == expected
    copied = copy_inputs(transferred)
    assert layout(copied) == expected and copied[0] is copied[2]
    assert copied[0].untyped_storage()._cdata == copied[1].untyped_storage()._cdata
    assert copied[0].untyped_storage()._cdata != transferred[0].untyped_storage()._cdata
    assert copied[3].dtype == torch.int64 and copied[4] is True
    before = copied[1].clone()
    copied[0].add_(5)
    assert not torch.equal(copied[1], before)
    assert torch.equal(base, torch.arange(80.).reshape(10, 8))


@pytest.mark.parametrize("shape,stride", [((3, 0, 4), (4, 4, 1)), ((3, 4), (0, 1)), ((), ())])
def test_empty_broadcast_and_scalar_view_metadata_survive_input_snapshots(shape, stride):
    base = torch.ones(8)
    value = base.as_strided(shape, stride, 2)
    copied = copy_inputs([value, value])
    assert layout(copied) == layout([value, value]) and copied[0] is copied[1]
    assert copied[0].untyped_storage()._cdata != base.untyped_storage()._cdata


def test_cross_dtype_aliases_are_explicitly_unsupported_instead_of_silently_split():
    tensor = torch.arange(8.)
    with pytest.raises(ValueError, match="Cross-dtype"):
        copy_inputs([tensor, tensor.view(torch.int32)])


def test_one_input_cannot_be_relabelled_as_a_shared_storage_delivery():
    with pytest.raises(ValueError, match="multiple uniform floating inputs"):
        input_layouts.expected("shared_storage", input_signature([torch.ones(2, 3)]))


def test_native_input_isolation_compares_real_alias_effects_in_both_orders():
    class Reference(torch.nn.Module):
        def forward(self, x, y):
            x.add_(1)
            return y * 2

    class Wrong(torch.nn.Module):
        def forward(self, x, y):
            x.clone().add_(1)
            return y * 2

    args = input_layouts.apply([torch.ones(2, 3), torch.ones(2, 3)], "shared_storage",
                              generator=None, allocation_device="cpu")
    saved = copy_inputs(args)
    for candidate, passed in ((Reference(), True), (Wrong(), False)):
        metadata = {}
        issue = check_paired_calls(Reference(), candidate, args, seed=42, tolerance=1e-6, device="cpu",
                                  set_seed=torch.manual_seed, metadata=metadata, label="alias")
        assert (issue is None) is passed
        record = metadata["reliability_checks"][0]
        assert record["input_layout"] == layout(args)
        assert all(item["layout"] == layout(args) for item in record["copied_layouts"])
    assert all(torch.equal(value, previous) for value, previous in zip(args, saved))


def test_storage_base_shortcut_cannot_pass_the_actual_nonzero_offset_case():
    class Wrong(torch.nn.Module):
        def forward(self, x):
            source = torch.empty(0, dtype=x.dtype).set_(x.untyped_storage(), 0, x.shape, x.stride())
            return torch.relu(source)

    source = torch.tensor([[-2., -1., 0., 1., 2.]])
    ordinary = {}
    assert check_paired_calls(torch.nn.ReLU(), Wrong(), [source], seed=42, tolerance=1e-6, device="cpu",
                             set_seed=torch.manual_seed, metadata=ordinary, label="zero-offset") is None
    args = input_layouts.apply([source], "storage_offset", generator=None, allocation_device="cpu")
    metadata = {}
    assert check_paired_calls(torch.nn.ReLU(), Wrong(), args, seed=42, tolerance=1e-6, device="cpu",
                             set_seed=torch.manual_seed, metadata=metadata, label="offset")
    assert metadata["reliability_checks"][0]["copied_layouts"][0]["layout"]["tensors"][0]["offset"] == 7


@pytest.mark.parametrize("pid", [26, 88])
def test_gelu_discriminator_exposes_replacing_the_declared_formula(pid):
    from kb_workflows.numerics.numerical import PrecisionContract, precision_gate
    from kb_workflows.numerics.operator_oracles import execute
    from kb_workflows.numerics.operator_specs import spec_for
    from tests.numerics.test_operator_oracles import small_case

    source, model, _ = small_case(pid)
    row = next(row for row in operator_cases.cases(pid, model, seed=42) if row["label"] == "gelu_discriminator")
    expected, truth = model(*row["inputs"]), execute(pid, model, row["inputs"])
    wrong = torch.nn.functional.gelu(row["inputs"][0], approximate="tanh" if pid == 26 else "none")
    contract = PrecisionContract(**spec_for(1, pid, source)["precision"]["float32"])
    assert precision_gate(expected, expected.clone(), truth, contract)["passed"]
    assert not precision_gate(expected, wrong, truth, contract)["passed"]


@pytest.mark.parametrize("pid", [41, 42, 43, 49, 51, 52, 53])
def test_border_extreme_is_unique_and_keeps_all_data_on_one_side_of_zero(pid):
    from tests.numerics.test_operator_oracles import small_case

    _, model, _ = small_case(pid)
    row = next(row for row in operator_cases.cases(pid, model, seed=42) if row["label"] == "isolated_extrema")
    value = row["inputs"][0]
    border = value.flatten()[0 if pid in {41, 42, 43} else -1]
    assert (value > 0).all() if pid in {52, 53} else (value < 0).all()
    assert (value == border).sum() == 1
    assert border == (value.min() if pid in {52, 53} else value.max())
    output = model(value)
    output = (output[0] if isinstance(output, tuple) else output).flatten()
    if pid in {51, 52}:
        assert output[-1] == value.shape[model.dim] - 1
    else:
        assert output[0 if pid in {41, 42, 43} else -1] == border


@pytest.mark.parametrize("pid", [*range(19, 33), 88])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_large_finite_activation_roles_stay_finite_at_the_stored_format(pid, dtype):
    from tests.numerics.test_operator_oracles import small_case

    _, model, _ = small_case(pid, dtype)
    row = next(row for row in operator_cases.cases(pid, model, seed=42) if row["label"] == "format_extremes")
    args = native._process_input_group(row["inputs"], "cpu", dtype)
    assert torch.isfinite(args[0]).all() and (args[0].abs() >= 32752).all()
    assert torch.isfinite(model(*args)).all()


@pytest.mark.parametrize("damage", ["summary", "source", "copy", "order", "count", "storage", "object", "dtype", "bool_id", "float_offset"])
def test_hidden_receipt_replay_requires_each_real_alias_delivery(tmp_path, monkeypatch, damage):
    from kb_workflows.numerics import operator_input_policy
    from tests.numerics.test_operator_inputs import execute_hidden_cpu

    request, result, _ = execute_hidden_cpu(tmp_path, monkeypatch)
    changed = copy.deepcopy(result)
    summary = next(row for row in changed["metadata"]["operator_inputs"]["cases"]
                   if row["family_label"] == "shared_storage")
    native = next(row for row in changed["metadata"]["reliability_checks"] if row["label"] == summary["label"])
    if damage == "summary":
        summary["input_layout"] = None
    elif damage == "source":
        native["input_layout"]["tensors"][0]["offset"] = 0
    elif damage == "copy":
        native["copied_layouts"][0]["layout"] = None
    elif damage == "order":
        native["copied_layouts"].reverse()
    elif damage == "count":
        native["copied_layouts"].pop()
    else:
        key, value = {"storage": ("storage", 1), "object": ("object", 1), "dtype": ("dtype", "torch.float64"),
                      "bool_id": ("object", False), "float_offset": ("offset", 7.)}[damage]
        native["copied_layouts"][0]["layout"]["tensors"][0][key] = value
    assert operator_input_policy.issue(changed, request) == "operator_input_family_alias_delivery_missing_or_changed"


@pytest.mark.parametrize("unsupported", [False, True])
def test_required_layout_cannot_qualify_after_backend_compaction_or_missing_storage_api(monkeypatch, unsupported):
    from kb_workflows.numerics import operator_input_policy

    args = input_layouts.apply([torch.ones(2, 3)], "storage_offset", generator=None, allocation_device="cpu")
    compacted = [args[0].clone()]
    token = operator_input_policy.ACTIVE.set({"records": []})
    try:
        if unsupported:
            monkeypatch.setattr("src.input_groups.layout", lambda value: None)
        with pytest.raises(ValueError, match="did not deliver"):
            operator_input_policy.completed({"label": "storage_offset"}, args if unsupported else compacted, True)
        assert not operator_input_policy.ACTIVE.get()["records"]
    finally:
        operator_input_policy.ACTIVE.reset(token)


@pytest.mark.parametrize("level,pid", [(2, 1), (2, 18), (3, 33), (3, 34), (3, 41)])
def test_actual_weight_ratios_target_cancellation_without_replacing_parameters(level, pid):
    from tests.support._composite_cases import small_composite_case

    if level == 2:
        source, model, args = small_composite_case(pid, dtype=torch.float32)
    else:
        source = problem_paths(3)[pid].read_text()
        namespace = {}
        Model, _, _ = native.load_original_model_and_inputs(source, namespace)
        namespace["batch_size"] = 2
        model = Model(3, 4, 2)
        args = ([torch.randn(2, 3), torch.randn(2, 4)] if pid == 33 else
                [torch.randn(5, 2, 3), torch.randn(2, 4)] if pid == 34 else
                [torch.randn(5, 2, 3), torch.randn(4, 2, 4)])
    spec = model_input_cases.reviewed_spec(level, pid, source)
    assert spec["stress_policy"]["direct_main"]
    values = copy_inputs(args)
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    receipt = model_stress.apply(values, "weight_cancellation", spec, reference=model, allocation_device="cpu")
    assert receipt["coefficient_source"] == "actual_reference_weight" and receipt["two_nonzero_products"]
    assert all(torch.equal(value, saved[name]) for name, value in model.state_dict().items())
    main = spec["stress_policy"]["direct_main"]
    layer = model.get_submodule(main["attribute"])
    weight = layer.weight_ih_l0 if main["kind"] == "recurrent" else layer.weight
    products = []
    for input_position, weight_position in zip(receipt["input_positions"][:2], receipt["weight_positions"][:2]):
        if len(input_position) == 1:
            data = values[0].reshape(-1, values[0].shape[-1])[0, input_position[0]]
        else:
            data = values[0][(0, *input_position)]
        products.append(float(data.double() * weight[tuple(weight_position)].double()))
    assert products[0] * products[1] < 0
    assert abs(sum(products)) < max(map(abs, products)) * 2e-7
    assert max(float(value.abs().max()) for value in values) <= 32


def test_ratio_construction_exposes_single_pass_bfloat_rounding_in_a_small_exact_fixture():
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.1234567, .2734567, .01654]]))
    spec = {"stress_policy": {"direct_main": {"kind": "linear", "attribute": "layer"}}}
    wrapper = torch.nn.Module()
    wrapper.layer = model
    args = [torch.zeros(1, 3)]
    model_stress.apply(args, "weight_cancellation", spec, reference=wrapper, allocation_device="cpu")
    truth = args[0].double() @ model.weight.detach().double().T
    rounded = (args[0].bfloat16().float() @ model.weight.detach().bfloat16().float().T).double()
    assert (rounded - truth).abs().item() > 1e-4 and truth.abs().item() < 1e-4


def test_alias_memory_admission_precedes_case_allocation(monkeypatch):
    spec = model_input_cases.reviewed_spec(3, 48, problem_paths(3)[48].read_text())
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: pytest.fail("Must admit before input allocation"))
    with pytest.raises(ValueError, match="memory envelope; no resizing"):
        next(model_input_cases.cases(spec, seed=42, memory_mb=1))


@pytest.mark.parametrize("damage", ["labels", "axis", "amplitude", "module"])
def test_model_stress_policy_is_validated_before_any_sdk_work(tmp_path, monkeypatch, damage):
    import json

    document = json.loads(model_input_cases.MANIFEST.read_text())
    policy = document["tasks"]["3/34"]["stress_policy"]
    if damage == "labels":
        policy["labels"].pop()
    elif damage == "axis":
        policy["feature_axes"][0] = True
    elif damage == "amplitude":
        policy["activation_amplitude"] = 65504
    else:
        policy["direct_main"]["kind"] = "unreviewed"
    path = tmp_path / "families.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(model_input_cases, "MANIFEST", path)
    with pytest.raises(ValueError, match="(reviewed|Incomplete)"):
        model_input_cases.reviewed_spec(3, 34, problem_paths(3)[34].read_text())

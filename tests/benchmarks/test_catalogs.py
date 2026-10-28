"""One catalog pass per level, reusing meta models across policy checks."""

import copy
import hashlib

import pytest
import torch

from kb_workflows.common import fingerprint, problem_paths
from kb_workflows.evaluation import baseline_rewrites, state_policies
from kb_workflows.evaluation.measurement import input_signature
from kb_workflows.evaluation.state_paths import plain_tensors
from kb_workflows.numerics import (
    input_layouts,
    input_profiles,
    model_input_cases,
    model_input_profiles,
    operator_cases,
    operator_input_policy,
)
from src import eval as native
from src.input_groups import copy_inputs, layout
from tests.support._model_cases import original_meta


@pytest.mark.parametrize("level", [1, 2, 3, 5])
def test_source_frozen_catalogs_and_meta_geometry(level):
    for pid, path in problem_paths(level).items():
        source = path.read_text()
        options = baseline_rewrites.extension_options(level, pid, source)
        assert baseline_rewrites.entry_for(level, pid, source)
        assert len(options) == len(set(options)) and len(options) % 2 == 0
        for name, plan in options.items():
            assert plan == baseline_rewrites.plan_for(level, pid, source, name)
            if not name.endswith("_cuda_graph"):
                counterpart = options[name + "_cuda_graph"]
                assert counterpart["rewrite"] == plan["rewrite"] and counterpart["backend_policy"] == plan["backend_policy"]
                assert counterpart["outer_graph"] and not plan["outer_graph"]
        policy = state_policies.plan(level, pid, source)
        assert policy["reference_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        with torch.no_grad():
            _, model, nominal = original_meta(level, pid, nominal=level in (2, 3))
            assert plain_tensors(model, allow_meta=True).keys() == policy["ordinary"].keys(), (level, pid)
            if level == 5:
                continue
            if level == 1:
                rows = list(operator_cases.cases(pid, model, seed=42, memory_mb=8192, allocation_device="meta"))
            else:
                spec = model_input_cases.reviewed_spec(level, pid, source)
                assert [list(value.shape) for value in nominal] == spec["nominal_shapes"]
                plan = operator_input_policy.plan(level, pid, source, 256)
                assert plan["reference_sha256"] == hashlib.sha256(source.encode()).hexdigest()
                assert plan["model_family_sha256"] == fingerprint(spec)
                assert plan["labels"] == list(model_input_cases.labels(spec))
                assert bool(plan["fixed_shape_reason"]) == ((level, pid) in {(3, 33), (3, 48), (3, 49)})
                value_policy = model_input_profiles.policy(level, pid, source)
                assert value_policy["arguments"] == len(nominal)
                assert tuple(value_policy["recipes"]) == input_profiles.recipes(pid, level)
                for recipe in value_policy["recipes"]:
                    values = input_profiles.transform(nominal, pid, recipe, level=level)
                    assert input_signature(values) == input_signature(nominal) and layout(values) == layout(nominal)
                rows = list(model_input_cases.cases(spec, seed=42, memory_mb=8192, allocation_device="meta", reference=model))
                assert [row["label"] for row in rows] == list(model_input_cases.labels(spec))
                from kb_workflows.numerics import composite_oracles, model_oracles
                from kb_workflows.numerics import specs as composite_specs
                from kb_workflows.numerics import specs as model_specs
                from kb_workflows.numerics.specs import spec_for

                oracle_spec = spec_for(level, pid, source)
                if level == 2:
                    prepared = composite_oracles.Prepared(model, nominal, expected_program=oracle_spec["program_sha256"])
                    assert prepared.description == composite_specs.composite_program(oracle_spec)
                    assert len(prepared.description["nodes"]) == oracle_spec["program_nodes"]
                else:
                    prepared = model_oracles.Prepared(model, nominal, problem_id=pid, expected_program=oracle_spec["program_sha256"])
                    assert prepared.description == model_specs.model_program(oracle_spec)
                assert prepared.estimated_host_bytes > 0
                assert all(value.device.type == "meta" for value in model.parameters())
            before = {name: (value.shape, value.dtype) for name, value in model.state_dict().items()}
            for row in rows:
                args = row["inputs"]
                tensors = [value for value in args if isinstance(value, torch.Tensor)]
                assert all(value.device.type == "meta" for value in tensors)
                if level in (2, 3):
                    assert [list(value.shape) for value in args] == model_input_cases.shapes(spec, row["label"])
                assert row["estimated_input_bytes"] == input_layouts.estimate(row["label"], [list(value.shape) for value in tensors])
                if row["label"] == "strided":
                    assert all(value.stride(-1) == 2 and not value.is_contiguous() for value in tensors)
                with torch.device("meta"):
                    output = copy.deepcopy(model)(*args)
                outputs = output if isinstance(output, (tuple, list)) else [output]
                assert all(value.device.type == "meta" for value in outputs)
            assert before == {name: (value.shape, value.dtype) for name, value in model.state_dict().items()}
            aliases = 0
            for row in rows:
                if row["label"] not in input_layouts.LABELS:
                    continue
                converted = native._process_input_group(row["inputs"], "meta", torch.bfloat16)
                expected = input_layouts.expected(row["label"], input_signature(converted))
                assert layout(converted) == layout(copy_inputs(converted)) == expected
                assert row["estimated_input_bytes"] == input_layouts.estimate(row["label"], [list(value.shape) for value in converted if isinstance(value, torch.Tensor)])
                aliases += 1
            assert aliases, (level, pid)


def test_catalog_source_drift_is_rejected():
    source = problem_paths(2)[1].read_text() + "\n# changed\n"
    for check in (baseline_rewrites.extension_options, state_policies.plan, model_input_profiles.policy):
        with pytest.raises(ValueError):
            check(2, 1, source)
    with pytest.raises(ValueError):
        operator_input_policy.plan(2, 1, source, 256)

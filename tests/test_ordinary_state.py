"""Ordinary state semantics on small CPU inputs; original sizes use meta only."""

import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from kb_workflows import state_audit, state_policies
from kb_workflows.common import ROOT, fingerprint, problem_paths
from kb_workflows.state_contract import state_checks
from kb_workflows.state_paths import plain_tensors
from src.eval import load_original_model_and_inputs


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def check(reference, candidate, args, *, policy=None):
    return state_checks(reference, candidate, args, seed=42, set_seed=torch.manual_seed,
                        device="cpu", tolerance=1e-5, policy=policy)


class Ordinary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        value = torch.tensor([1., 2., 3.])
        self.state = {"values": [value, value], 1: torch.tensor(0)}

    def forward(self, x):
        if self.training:
            self.state["values"][0].add_(1)
            self.state[1].add_(1)
        return x + self.state["values"][0]


def test_builtin_paths_keep_key_values_and_types_and_tied_tensor_identity():
    reference = Ordinary()
    candidate = copy.deepcopy(reference)
    candidate.state = {1: candidate.state[1], "values": candidate.state["values"]}
    result = check(reference, candidate, [torch.randn(2, 3)])
    assert result["status"] == "passed", result
    states = result["values"]["bindings"][0]["states"]
    assert {row["reference"] for row in states} == {'@state["values"][0]', '@state["values"][1]', '@state[1]'}
    aliases = {row["reference"]: row["alias_group"] for row in states}
    assert aliases['@state["values"][0]'] == aliases['@state["values"][1]']
    candidate.state["values"][1] = candidate.state["values"][1].clone()
    assert check(reference, candidate, [torch.randn(2, 3)])["status"] == "failed"


def test_replaced_ordinary_state_is_compared_after_every_call():
    class Persistent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.ones(3)

        def forward(self, x):
            self.hidden = self.hidden + x.sum(0)
            return self.hidden

    reference = Persistent()
    assert check(reference, copy.deepcopy(reference), [torch.randn(2, 3)])["status"] == "passed"

    class Broken(Persistent):
        def forward(self, x):
            output = super().forward(x).clone()
            self.hidden.zero_()
            return output

    result = check(reference, Broken(), [torch.ones(2, 3)])
    assert result["status"] == "failed" and any("state.@hidden" in issue for issue in result["values"]["issues"])


def test_ordinary_reference_can_map_to_registered_candidate_buffer():
    class Reference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.ones(3)

        def forward(self, x):
            self.hidden = self.hidden + x.sum(0)
            return self.hidden

    class Candidate(torch.nn.Module):
        kernelbench_state_map = {"@hidden": "running"}

        def __init__(self):
            super().__init__()
            self.register_buffer("running", torch.ones(3))

        def forward(self, x):
            self.running = self.running + x.sum(0)
            return self.running

    assert check(Reference(), Candidate(), [torch.randn(2, 3)])["status"] == "passed"


def test_plain_object_namespaces_are_snapshotted_without_reading_properties():
    calls = []

    class State:
        def __init__(self):
            self.value = torch.ones(3)

        @property
        def misleading(self):
            calls.append(True)
            raise AssertionError("A state property must not execute")

    class Reference(torch.nn.Identity):
        def __init__(self):
            super().__init__()
            self.cache = SimpleNamespace(inner=State())

    reference = Reference()
    assert set(plain_tensors(reference)) == {"@cache.inner.value"}
    assert check(reference, copy.deepcopy(reference), [torch.ones(2, 3)])["status"] == "passed"
    assert calls == []


@pytest.mark.parametrize("bad", ["values", "shared", "removed"])
def test_snapshot_cannot_change_or_reuse_ordinary_state(bad):
    class Candidate(Ordinary):
        def __deepcopy__(self, memo):
            result = Ordinary()
            if bad == "values":
                result.state["values"][0].zero_()
            elif bad == "shared":
                result.state = self.state
            else:
                result.state = {}
            return result

    result = check(Ordinary(), Candidate(), [torch.randn(2, 3)])
    assert result["status"] == "failed" and any("snapshot" in issue for issue in result["values"]["issues"])


@pytest.mark.parametrize("kind", ["gru", "lstm", "rnn"])
def test_framework_recurrent_parameter_caches_are_not_extra_plain_state(kind):
    layer = {"gru": torch.nn.GRU, "lstm": torch.nn.LSTM, "rnn": torch.nn.RNN}[kind](3, 4, 2)
    assert plain_tensors(layer) == {}
    assert check(layer, copy.deepcopy(layer), [torch.randn(5, 2, 3)])["status"] == "passed"


@pytest.mark.parametrize("pid", [33, 41])
def test_source_bound_recurrent_semantics_on_explicit_small_native_models(pid):
    source = problem_paths(3)[pid].read_text()
    namespace = {}
    Model, _, _ = load_original_model_and_inputs(source, namespace)
    namespace["batch_size"] = 2  # Unit-test configuration; not default-task evidence.
    reference = Model(*([3, 4, 2] if pid == 33 else [3, 4, 2]))
    candidate = copy.deepcopy(reference)
    args = [torch.randn(2, 3), torch.randn(2, 4)] if pid == 33 else [torch.randn(5, 2, 3), torch.randn(4, 2, 4)]
    policy = state_policies.plan(3, pid, source)
    if pid == 41:
        del candidate.h0  # The reference forward uses its argument, not self.h0.
    result = check(reference, candidate, args, policy=policy)
    assert result["status"] == "passed", result
    assert state_audit._check_valid(result)
    ignored = result["values"]["bindings"][0]["ignored_ordinary"]
    assert bool(ignored) is (pid == 41)
    if pid == 33:
        assert "@hidden" in {row["reference"] for row in result["values"]["bindings"][0]["states"]}
        candidate.hidden.zero_()
        assert check(reference, candidate, args, policy=policy)["status"] == "failed"


@pytest.mark.parametrize("level,pid", [(level, pid) for level, count in ((1, 100), (2, 100), (3, 50), (5, 41)) for pid in range(1, count + 1)])
def test_task_policies_bind_current_sources_and_original_plain_state_metadata(level, pid):
    source = problem_paths(level)[pid].read_text()
    policy = state_policies.plan(level, pid, source)
    assert policy["reference_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    with pytest.raises(ValueError):
        state_policies.plan(level, pid, source + "\n# edited\n")
    linspace = torch.linspace

    def constants(*args, **kwargs):
        assert len(args) >= 3 and args[2] <= 4096
        return linspace(*args, **{**kwargs, "device": "cpu"})

    # Constructors only: never call nominal factories (one legacy factory has
    # a hardcoded .cuda()). Swin's bounded stochastic-depth constants use CPU.
    with torch.device("meta"), patch.object(torch, "linspace", constants), patch.object(
            torch.Tensor, "cuda", side_effect=AssertionError("No CUDA calls in metadata tests")):
        Model, init, _ = load_original_model_and_inputs(source, {})
        model = Model(*init())
    ordinary = plain_tensors(model, allow_meta=True)
    assert ordinary.keys() == policy["ordinary"].keys()


def test_module_modes_and_grad_postconditions_are_replayed_from_all_bound_owners():
    reference = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    record = check(reference, copy.deepcopy(reference), [torch.randn(5, 3)])
    assert record["status"] == "passed" and state_audit._check_valid(record)
    modes = record["values"]["events"][0]["module_modes_before"]
    assert {row["reference"] for row in modes} == {"", "0", "1"}
    record["values"]["events"][0]["module_modes_before"] = modes[:1]
    record["values"]["events"][0]["module_modes_after"] = modes[:1]
    assert not state_audit._check_valid(record)


def test_source_policy_previews_and_payloads_stay_sdk_light_and_queue_validates(tmp_path):
    from kb_workflows import engine
    from kb_workflows.common import kernel_name
    from kb_workflows.config import RunConfig
    from kb_workflows.queue_worker import validate_payload

    config = RunConfig(generation="local", state_validation=True, level=3, start=41, end=41, runs_dir=str(tmp_path)).validate()
    config.run_dir.mkdir(parents=True)
    (config.run_dir / kernel_name(3, 41)).write_text("candidate")
    request = engine.task_payload(config, 41)
    validate_payload(request)
    before = fingerprint(request)
    request["eval_config"]["state_policy"]["ordinary"]["@h0"]["role"] = "persistent"
    assert fingerprint(request) != before
    with pytest.raises(ValueError, match="task-state policy"):
        validate_payload(request)
    script = f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from kb_workflows.cli import main
assert main(['--state-validation','--level','3','--start','33','--end','41','--dry-run']) == 0
assert not {{'torch','triton','nki','torch_neuronx'}} & sys.modules.keys()
'''
    assert json.loads(subprocess.run([sys.executable, "-I", "-c", script], check=True, capture_output=True, text=True).stdout)["config"]["state_validation"]


@pytest.mark.parametrize("tamper", ["policy", "ignored", "missing_modes", "grad", "mode_value", "owner"])
def test_native_state_receipt_replays_task_policy_ignored_roles_and_module_modes(tamper):
    reference = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    collector = state_audit.Collector()
    collector.observe(reference, copy.deepcopy(reference), [torch.randn(5, 3)], seed=42, device="cpu",
                      set_seed=torch.manual_seed, tolerance=1e-5, label="native")
    document = collector.document()
    cases = [{"label": "native", "seed": 42}]
    assert state_audit.verdict(document, cases)["status"] == "passed"
    values = document["cases"][0]["check"]["values"]
    if tamper == "policy":
        document["policy"] = {"ordinary": {}}
    elif tamper == "ignored":
        values["bindings"][0]["ignored_ordinary"] = [{"reference": "@h0"}]
    elif tamper == "missing_modes":
        values["events"][0]["module_modes_before"].pop()
        values["events"][0]["module_modes_after"].pop()
    elif tamper == "grad":
        values["events"][0]["no_grad_after"] = False
    elif tamper == "mode_value":
        values["events"][0]["module_modes_after"][1]["actual"] = False
    else:
        binding = values["bindings"][0]
        binding["states"][0]["reference_owner"] = "missing"
        binding["sha256"] = hashlib.sha256(json.dumps(binding["states"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert state_audit.verdict(document, cases)["status"] == "not_measured"


@pytest.mark.parametrize("kind", ["cycle", "slotted", "dict_subclass", "unregistered_module"])
def test_unsupported_ordinary_state_is_explicit_and_never_silently_ignored(kind):
    class Slotted:
        __slots__ = ("tensor",)

        def __init__(self):
            self.tensor = torch.ones(3)

    class Custom(dict):
        pass

    model = torch.nn.Identity()
    if kind == "cycle":
        value = []
        value.append(value)
    elif kind == "slotted":
        value = Slotted()
    elif kind == "dict_subclass":
        value = Custom(tensor=torch.ones(3))
    else:
        value = [torch.nn.Linear(3, 3)]
    model.payload = value
    assert check(model, copy.deepcopy(model), [torch.ones(2, 3)])["status"] == "failed"


@pytest.mark.parametrize("alias", ["tie", "cache"])
def test_custom_snapshot_cannot_repair_a_broken_alias_or_reconnect_a_stale_cache(alias):
    class Candidate(torch.nn.Linear):
        def __init__(self):
            super().__init__(3, 3, bias=False)
            if alias == "tie":
                self.other = torch.nn.Parameter(self.weight.detach().clone())
            else:
                self.cache = self.weight.detach().clone()

        def __deepcopy__(self, memo):
            result = Candidate()
            result.weight.data.copy_(self.weight)
            if alias == "tie":
                result.other = result.weight
            else:
                result.cache = result.weight.detach()
            return result

        def forward(self, x):
            return x @ (self.other if alias == "tie" else self.cache)

    class Reference(torch.nn.Linear):
        def __init__(self):
            super().__init__(3, 3, bias=False)
            if alias == "tie":
                self.other = self.weight

        def forward(self, x):
            return x @ self.weight

    reference, candidate = Reference(), Candidate()
    candidate.weight.data.copy_(reference.weight)
    if alias == "tie":
        candidate.other.data.copy_(reference.weight)
    else:
        candidate.cache.copy_(reference.weight)
    result = check(reference, candidate, [torch.randn(2, 3)])
    assert result["status"] == "failed"
    assert any("snapshot changed tensor object aliases" in issue for issue in result["values"]["issues"])

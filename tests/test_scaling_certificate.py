"""Finite-regime certificate math and explicit instances, without device measurements."""

import copy
import hashlib
import shutil
from types import SimpleNamespace

import pytest

from kb_workflows import engine, final_verification, production
from kb_workflows import scaling_certificate as scaling
from kb_workflows.common import (
    ROOT,
    atomic_json,
    evaluator_digest,
    fingerprint,
    kernel_name,
    read_json,
    select_problems,
)
from kb_workflows.config import RunConfig, parse_config


def points(candidate=1.0, reference=2.0):
    return [{"size": n, "groups": [[{"reference_ms": n**reference, "candidate_ms": n**candidate,
             "control_ms": n**reference} for _ in range(10)] for _ in range(2)]} for n in (16, 32, 64, 128, 256)]


def test_scaling_credits_a_verified_exponent_gap_not_a_constant_factor():
    observed = scaling.fit(points(), candidate_exponent=1, reference_exponent=2, tolerance=0.05, trials=200)
    assert observed["passed"] and observed["candidate_exponent"] == pytest.approx(1)
    constant = points(candidate=2)
    for point in constant:
        for group in point["groups"]:
            for row in group:
                row["candidate_ms"] /= 100
    observed = scaling.fit(constant, candidate_exponent=1, reference_exponent=2, tolerance=0.05, trials=200)
    assert not observed["passed"] and not observed["matches_reviewed_regime"]


@pytest.mark.parametrize("damage", ["few_sizes", "span", "duplicate", "one_repeat", "few_pairs", "nonfinite"])
def test_scaling_rejects_insufficient_or_invalid_evidence(damage):
    values = points()
    if damage == "few_sizes":
        values.pop()
    elif damage == "span":
        values[-1]["size"] = 255
    elif damage == "duplicate":
        values[-1]["size"] = values[0]["size"]
    elif damage == "one_repeat":
        values[0]["groups"].pop()
    elif damage == "few_pairs":
        values[0]["groups"][0].pop()
    else:
        values[0]["groups"][0][0]["candidate_ms"] = float("nan")
    with pytest.raises(ValueError):
        scaling.fit(values, candidate_exponent=1, reference_exponent=2, tolerance=0.1, trials=200)


def test_scaling_checks_every_shape_dtype_and_unscaled_input():
    contract = {"axis": {"input": 0, "dimension": 1}, "shape_laws": [{"input": 0, "dimension": 1, "power": 1}]}
    values = [[{"shape": [2, n], "dtype": "torch.float32"}, {"type": "int", "value": 3}] for n in (16, 32, 64, 128, 256)]
    assert scaling.shape_regime(values, contract) == [16, 32, 64, 128, 256]
    changed = copy.deepcopy(values)
    changed[1][0]["shape"][0] = 4
    with pytest.raises(ValueError, match="reviewed scaling regime"):
        scaling.shape_regime(changed, contract)
    changed = copy.deepcopy(values)
    changed[1][1]["value"] = 5
    with pytest.raises(ValueError, match="non-tensor"):
        scaling.shape_regime(changed, contract)
    malformed = copy.deepcopy(contract)
    malformed["axis"]["input"] = 100
    with pytest.raises(ValueError, match="Scaling axis"):
        scaling.shape_regime(values, malformed)


def test_explicit_production_dimensions_preserve_roles_and_private_hidden_cases():
    from kb_workflows.production_specs import task

    spec = task(8)
    with production.production_context(profile="small", precision="fp32", memory_bytes=64 * 1024**2,
                                       dimensions_override={"S": 16, "L": 16}):
        nominal = production.make_inputs(8)
        hidden = production.make_inputs(8, case="hidden_odd")
    assert nominal[0].shape[-1] == 16
    assert hidden[0].shape[-1] == 19 and hidden[1].shape[-1] == 9
    assert len(nominal) == len(spec["roles"])
    with pytest.raises(ValueError, match="silent resizing"):
        production.instance_plan(8, dimensions_override={"S": 1_000_000, "L": 1_000_000}, memory_bytes=1024)
    with pytest.raises(ValueError, match="divisibility"):
        production.instance_plan(8, dimensions_override={"D": 3})


def test_dimension_file_changes_payload_and_frozen_final_plan(tmp_path):
    from kb_workflows.generation import generation_signature

    dims = tmp_path / "dimensions.json"
    atomic_json(dims, {"S": 16, "L": 16})
    config = RunConfig(level=5, suite="production41", start=8, end=8, task_profile="small", task_dimensions=str(dims),
                       run_name="dimensions", runs_dir=str(tmp_path), hardware="RTX_A6000", final_repeats=2, resume=True).validate()
    config.run_dir.mkdir()
    (config.run_dir / kernel_name(5, 8)).write_text("class ModelNew: pass")
    before = fingerprint(engine.task_payload(config, 8))
    generation_before = generation_signature(config, 8, 0)
    manifest = final_verification.freeze(config)
    atomic_json(dims, {"S": 32, "L": 32})
    assert fingerprint(engine.task_payload(config, 8)) != before
    assert generation_signature(config, 8, 0) != generation_before
    assert final_verification.freeze(config)["signature"] != manifest["signature"]
    toml = tmp_path / "dimensions.toml"
    toml.write_text('[run]\nlevel=5\nsuite="production41"\ntask_dimensions="dimensions.json"\n')
    assert parse_config(["--config", str(toml)]).task_dimensions == str(dims)


def test_dimension_input_is_private_in_harbor(tmp_path):
    from kb_workflows.harbor import export_tasks

    dims = tmp_path / "dimensions.json"
    atomic_json(dims, {"S": 16})
    config = RunConfig(level=5, suite="production41", start=1, end=1, task_dimensions=str(dims), run_name="harbor-dims",
                       runs_dir=str(tmp_path), hardware="RTX_A6000", stage="export-harbor").validate()
    path = export_tasks(config) / "l5-p1-s0"
    assert (path / "tests/task_dimensions.json").is_file()
    assert not (path / "environment/task_dimensions.json").exists()


@pytest.fixture
def simulated_scaling(tmp_path, monkeypatch):
    """Native certificate/scoring logic receives simulated timing and correctness evidence."""
    from kb_workflows import calibration, headroom, paper_analysis, rl_plan, rl_rewards
    from kb_workflows.baseline_variants import VARIANTS, select_baseline
    from kb_workflows.final_verification import PROTOCOL as FINAL_PROTOCOL
    from kb_workflows.measurement import timing_stats
    from kb_workflows.paired_statistics import paired_summary

    device = {"available": True, "name": "SIMULATED", "uuid": "simulated-uuid", "driver_version": "fixture",
              "torch_version": "fixture", "cuda_version": "fixture", "hip_version": None, "capability": [8, 6],
              "total_memory": 10**10, "multiprocessors": 10, "l2_bytes": 1000}
    root = tmp_path / "verifier"
    for pid in (3, 8):
        hidden = root / "hidden_tests/level5" / f"{pid}_hidden.py"
        hidden.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "hidden_tests/level5" / f"{pid}_hidden.py", hidden)
    for module in (engine, paper_analysis, rl_rewards):
        monkeypatch.setattr(module, "ROOT", root)

    def signature(config):
        overrides = production.read_dimensions(config.task_dimensions) if config.task_dimensions else None
        instance = production.instance_plan(8, profile="small", dimensions_override=overrides)
        return [{"shape": row["shape"], "dtype": "torch.float32", "stride": [1] * len(row["shape"])} for row in instance["input_roles"]]

    dims = tmp_path / "anchor.json"
    atomic_json(dims, {"S": 16, "L": 16})
    metadata = {"device_fingerprint": device, "precision": "fp32", "input_signature": signature(SimpleNamespace(task_dimensions=str(dims)))}
    calibrated = calibration.seal(device, {"memory_bytes": calibration.probe_record(1000, [1] * 5, unit="bytes/s", parameters={})},
                                  [0.001] * 5, settings={"fixture": "not a device experiment"})
    reference_sha = hashlib.sha256(select_problems(5, problem_ids=[8])[8].read_bytes()).hexdigest()
    groups = []
    for repeat in (0, 1):
        groups.append({phase: {"compiled": True, "correctness": True, "runtime": 0.2, "runtime_stats": {"samples_ms": [0.2, 0.2]},
            "metadata": {**copy.deepcopy(metadata), "kernel_sha256": "d" * 64, "reference_sha256": "e" * 64,
                "evaluation_config": {"evaluator_sha256": evaluator_digest(), "final_protocol": FINAL_PROTOCOL,
                                      "final_nonce": f"independent-{repeat}", "use_hidden_tests": phase == "hidden"}}}
                       for phase in ("standard", "hidden")})
    witness = {"protocol": "independent_reachability_v1", "correctness": True, "hidden_correctness": True,
        "device_identity_sha256": calibration.identity(device), "calibration_sha256": calibrated["receipt_sha256"],
        "reference_sha256": reference_sha, "source_sha256": "c" * 64, "source_fingerprint": "d" * 64, "reference_fingerprint": "e" * 64,
        "input_signature": metadata["input_signature"], "precision": "fp32", "runtime_ms": 0.2,
        "verification": groups, "verification_receipt_sha256": fingerprint({"groups": groups})}
    regime = {"axis": {"input": 0, "dimension": 2}, "shape_laws": [
        {"input": 0, "dimension": 2, "power": 1}, {"input": 1, "dimension": 1, "power": 1}],
        "candidate_exponent": 1.2, "reference_exponent": 2.0, "tolerance": 0.05}
    bound = {"protocol": headroom.PROTOCOL, "layer": "D", "reference_sha256": reference_sha, "precision": "fp32",
        "input_signature": metadata["input_signature"], "review": {"reviewer": "fixture", "rationale": "Synthetic software contract"},
        "calibration_sha256": calibrated["receipt_sha256"], "reference_work": {"memory_bytes": 2560}, "algorithm_work": {"memory_bytes": 100},
        "design": {"algorithm": "fixture", "comparison": "fixture", "numerical_contract": "fixture", "maximum_witness_gap": 3,
                   "reachability": witness, "scaling": regime}}
    contract = {**regime, "reference_sha256": reference_sha, "bound_certificate_sha256": fingerprint(bound),
        "review": {"reviewer": "fixture", "rationale": "Synthetic finite-range scaling regime"},
        "instances": [{"dimensions": {"S": n, "L": n}, "memory_mb": 64} for n in (16, 32, 64, 128, 256)], "baseline_variant": "eager"}
    specs, calibration_path, contracts = tmp_path / "specs.json", tmp_path / "calibration.json", tmp_path / "scaling.json"
    atomic_json(specs, {"level5": {"8": {"layer": "D", "headroom_certificate": bound}}})
    atomic_json(calibration_path, calibrated)
    atomic_json(contracts, {"protocol": scaling.CONTRACT_PROTOCOL, "tasks": {"5/8": contract}})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    plan = rl_plan.make_plan({"training": {"checkpoint": str(checkpoint), "group_size": 2, "certificate_bonus": 0.05,
        "certificate_min_score": 0.1, "scaling_contracts": str(contracts)},
        "verifier_config": {"level": 5, "suite": "production41", "task_profile": "small", "task_dimensions": str(dims),
            "score_mode": "headroom", "problem_specs": str(specs), "device_calibration": str(calibration_path),
            "strict_validation": False, "evidence_level": "current", "hardware": "RTX_A6000", "final_repeats": 2,
            "paired_trials": 10, "num_perf_trials": 3}, "train": [{"level": 5, "problem_id": 8}], "heldout": [{"level": 5, "problem_id": 3}]},
        directory=tmp_path / "training")
    timing_receipt = {"output_consumed": True, "fresh_inputs": True, "varied_input_contents": True,
                      "timed_outputs_verified": True, "input_seeds": [1, 2, 3]}

    def times(config):
        n = signature(config)[0]["shape"][2]
        return n**2 * 0.01, n**1.2 * 0.01

    def baseline(config):
        variants = {}
        ref, _ = times(config)
        for name in VARIANTS:
            payload = engine.task_payload(config, 8, task="baseline")
            payload["eval_config"]["baseline_variant"] = name
            variants[name] = {**timing_stats([ref] * 3), "baseline_correctness": True, "precision": "fp32", "tf32": False,
                "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}), "input_signature": signature(config),
                "device_fingerprint": device, "cold_start_ms": 0.01, "timing_method": "synchronized_wall", "timing_receipt": timing_receipt,
                "metadata": {"evaluation_config": payload["eval_config"], "workflow_signature": fingerprint(payload)}}
        value = select_baseline(variants)
        value["suite_complete"] = True
        value["metadata"]["workflow_signature"] = fingerprint(engine.task_payload(config, 8, task="baseline"))
        reference = select_problems(5, problem_ids=[8])[8]
        atomic_json(root / "results/timing/RTX_A6000" / f"{config.baseline}.json", {"level5": {reference.name: value}})

    def dispatch(config, payload):
        ref, candidate = times(config)
        good = "WRONG" not in payload["kernel_src"]
        result = {"compiled": True, "correctness": good, "runtime": candidate, "runtime_stats": timing_stats([candidate] * 3),
            "metadata": {**copy.deepcopy(metadata), "input_signature": signature(config), "device_fingerprint": device,
                "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}), "kernel_sha256": fingerprint({"source": payload["kernel_src"]}),
                "cold_start_ms": 0.01, "timing_method": "synchronized_wall", "timing_receipt": timing_receipt}}
        if payload["task"] == "paired_timing":
            pairs = [{"reference_ms": ref, "candidate_ms": candidate, "control_ms": ref} for _ in range(10)]
            result["metadata"]["paired_timing"] = {**paired_summary(pairs), "baseline_variant": payload["eval_config"]["paired_baseline_variant"],
                "output_consumed": True, "varied_inputs": True, "all_outputs_verified": True}
        return result
    anchor = rl_plan.task_config(plan, "5/8", directory=tmp_path / "anchor")
    baseline(anchor)
    return plan, contract, metadata, dispatch, baseline


def test_full_scaling_collector_uses_frozen_source_and_native_evidence(simulated_scaling, tmp_path):
    plan, contract, metadata, dispatch, baseline = simulated_scaling
    source = "import torch\nclass ModelNew(torch.nn.Module):\n    def forward(self, *args): return args[0]\n"
    receipt = scaling.collect(plan, "5/8", source, contract, directory=tmp_path / "certificate", anchor_metadata=metadata,
                              dispatch=dispatch, baseline_recorder=baseline)
    assert receipt["fit"]["passed"] and len(receipt["points"]) == 5
    assert scaling.validate(tmp_path / "certificate/certificate.json")["fit"]["passed"]
    original = {item["directory"]: read_json(tmp_path / "certificate" / item["directory"] / "run_config.json")["baseline"]
                for item in receipt["points"]}
    second = scaling.collect(plan, "5/8", source, contract, directory=tmp_path / "second", anchor_metadata=metadata,
                             dispatch=dispatch, baseline_recorder=baseline)
    assert not set(original.values()) & {read_json(tmp_path / "second" / item["directory"] / "run_config.json")["baseline"] for item in second["points"]}
    assert scaling.validate(tmp_path / "certificate/certificate.json")["fit"]["passed"]
    path = tmp_path / "certificate/case-0/evaluation/scaling_pairs.json"
    atomic_json(path, {})
    with pytest.raises(ValueError, match="evidence changed"):
        scaling.validate(tmp_path / "certificate/certificate.json")


def test_bonus_is_bounded_and_unavailable_to_a_failed_candidate(simulated_scaling, tmp_path):
    from kb_workflows import rl_rewards

    plan, contract, metadata, dispatch, baseline = simulated_scaling
    source = "import torch\nclass ModelNew(torch.nn.Module):\n    def forward(self, *args): return args[0]\n"
    calls = []
    def collector(*args, **kwargs):
        calls.append(args)
        return scaling.collect(*args, **kwargs, baseline_recorder=baseline)
    result = rl_rewards.evaluate_group(plan, "5/8", [source, source + "\n# WRONG\n"], directory=tmp_path / "reward",
                                      dispatch=dispatch, certificate_collector=collector)
    assert len(calls) == 1 and result["rewards"][1] == 0
    assert result["components"][0]["certificate_bonus"] == pytest.approx(0.05)
    assert result["rewards"][0] == result["components"][0]["qualified"] + 0.05 < 1
    assert rl_rewards.validate_reward(tmp_path / "reward/reward.json")["rewards"] == result["rewards"]
    changed = copy.deepcopy(contract)
    changed["candidate_exponent"] = 0.5
    with pytest.raises(ValueError, match="reviewed D/SOL"):
        scaling.bind_contract(plan, "5/8", changed, metadata)

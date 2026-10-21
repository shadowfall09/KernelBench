"""Production reference, role, numerical and integration regressions on CPU only."""

import copy
import importlib.util
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from kb_workflows import engine, generation, production, production_ops, suites
from kb_workflows.common import ROOT, fingerprint, kernel_name, read_json, select_problems
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.production_specs import task
from src.eval import _process_input_tensor, input_dtype_policy, load_original_model_and_inputs, move_model


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tensors(value):
    return value if isinstance(value, tuple) else (value,)


@pytest.mark.parametrize("pid", range(1, 42))
def test_every_family_has_explicit_roles_and_finite_dtype_instances(pid):
    spec = task(pid)
    for precision in spec["dtype_families"]:
        args = production.make_inputs(pid, precision=precision, profile="small", seed=123)
        plan = production.instance_plan(pid, precision=precision, profile="small")
        for row, value in zip(plan["input_roles"], args):
            if row["dtype"] == "scalar":
                continue
            assert list(value.shape) == row["shape"]
            actual = str(value.dtype).replace("torch.", "")
            expected = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "fp8_e4m3fn": "float8_e4m3fn"}.get(row["dtype"], row["dtype"])
            assert actual == expected
        result = production.execute(pid, *args)
        assert all(torch.isfinite(value).all() for value in tensors(result))
        if spec["gradient_indices"]:
            assert len(result) == len(spec["gradient_indices"])
            for index, gradient in zip(spec["gradient_indices"], result):
                assert gradient.shape == args[index].shape and gradient.dtype == args[index].dtype


@pytest.mark.parametrize("pid", range(1, 42))
def test_hidden_odd_shapes_strides_and_domains_preserve_legality(pid):
    odd = production.make_inputs(pid, case="hidden_odd", seed=123)
    strided = production.make_inputs(pid, case="hidden_strided", seed=123)
    for left, right in zip(odd, strided):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left.float(), right.float(), rtol=0, atol=0)
    assert any(isinstance(value, torch.Tensor) and value.ndim and not value.is_contiguous() for value in strided)
    expected, actual = production.execute(pid, *odd), production.execute(pid, *strided)
    for left, right in zip(tensors(expected), tensors(actual)):
        torch.testing.assert_close(left, right)
    extreme = production.make_inputs(pid, case="hidden_extreme", seed=123)
    assert all(torch.isfinite(value).all() for value in tensors(production.execute(pid, *extreme)))


@pytest.mark.parametrize("pid", range(31, 42))
def test_each_backward_vjp_matches_fp64_directional_finite_difference(pid):
    spec = task(pid)
    args = production.make_inputs(pid, profile="small", seed=12)
    args = [value.double() if isinstance(value, torch.Tensor) and value.is_floating_point() else value for value in args]
    gradients = production.execute(pid, *args)
    function = getattr(production_ops, spec["operation"])
    differentiable = [args[index].detach().requires_grad_(True) for index in spec["gradient_indices"]]
    def gradient_reference(*values):
        forward_args = list(args[:-1])
        for index, value in zip(spec["gradient_indices"], values):
            forward_args[index] = value
        return function(*forward_args)
    assert torch.autograd.gradcheck(gradient_reference, tuple(differentiable), fast_mode=True, atol=1e-4, rtol=1e-3)
    generator = torch.Generator().manual_seed(19)
    step = 1e-5
    for index, gradient in zip(spec["gradient_indices"], gradients):
        direction = torch.randn(args[index].shape, dtype=torch.float64, generator=generator)
        plus, minus = list(args[:-1]), list(args[:-1])
        plus[index], minus[index] = args[index] + step * direction, args[index] - step * direction
        finite = ((function(*plus) - function(*minus)) * args[-1]).sum() / (2 * step)
        predicted = (gradient * direction).sum()
        torch.testing.assert_close(predicted, finite, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("pid", [1, 2, 3, 4, 8, 19, 23, 27, 28, 30])
def test_mathematical_references_against_independent_torch_formulations(pid):
    args = production.make_inputs(pid, profile="small", seed=3)
    actual = production.execute(pid, *args)
    if pid == 1:
        q, k, v, mask = args
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    elif pid == 2:
        x, weight, labels = args
        expected = F.cross_entropy(F.linear(x, weight), labels)
    elif pid == 3:
        x, weight, bias, eps = args
        expected = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    elif pid == 4:
        x, count = args
        expected = torch.topk(x, count, dim=-1)
    elif pid == 8:
        x, kernel = args
        full = torch.fft.irfft(torch.fft.rfft(x, n=x.shape[-1] + kernel.shape[-1] - 1)
            * torch.fft.rfft(kernel, n=x.shape[-1] + kernel.shape[-1] - 1)[None], n=x.shape[-1] + kernel.shape[-1] - 1)
        expected = full[..., :x.shape[-1]]
    elif pid == 19:
        expected = (args[0].float() @ args[1].float().T).bfloat16()
    elif pid == 23:
        expected = torch.stack([left @ right.T for left, right in zip(*args)])
    elif pid == 27:
        x, residual, weight, eps = args
        expected = F.rms_norm(x + residual, (x.shape[-1],), weight, eps)
    elif pid == 28:
        x, cosine, sine = args
        pair = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        rotation = torch.complex(cosine, sine)
        expected = torch.view_as_real(pair * rotation).flatten(-2)
    else:
        q, k, v = args
        expected = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1))
    for left, right in zip(tensors(actual), tensors(expected)):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_packed_int4_order_sign_scales_and_stable_topk_ties():
    packed = torch.tensor([[0xF0, 0x87]], dtype=torch.uint8)
    assert production_ops.unpack_int4(packed, torch.tensor([[2.0, 3.0]]), 2).tolist() == [[-16, 14, -3, 0]]
    values, indices = production_ops.partial_topk(torch.tensor([[2.0, 2.0, 1.0, 2.0]]), 3)
    assert indices.tolist() == [[0, 1, 3]] and values.tolist() == [[2, 2, 2]]
    quantized = production_ops.block_fp4_gemm(torch.tensor([[0.75, 1.25, 2.5, 6.0]]), torch.eye(4), 4)
    assert quantized.tolist() == [[1.0, 1.0, 2.0, 6.0]]  # E2M1 ties to even magnitude codes.


def test_role_policy_preserves_fp32_scales_fp8_storage_and_buffers():
    scale = torch.tensor([0.125], dtype=torch.float32)
    packed = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
    assert _process_input_tensor(scale, "cpu", torch.bfloat16).dtype == torch.bfloat16
    model = torch.nn.Module()
    model.register_buffer("scale", scale)
    with input_dtype_policy(preserve=True):
        assert _process_input_tensor(scale, "cpu", torch.bfloat16).dtype == torch.float32
        assert _process_input_tensor(packed, "cpu", torch.bfloat16).dtype == torch.float8_e4m3fn
        assert move_model(model, "cpu", torch.bfloat16).scale.dtype == torch.float32
    assert _process_input_tensor(scale, "cpu", torch.bfloat16).dtype == torch.bfloat16


def test_memory_scaling_is_bounded_replayable_and_capacity_aware():
    for pid in range(1, 42):
        plan = production.instance_plan(pid, profile="large", memory_bytes=2**20)
        assert plan["estimated_peak_bytes"] <= plan["memory_budget_bytes"]
        assert plan == production.instance_plan(pid, profile="large", memory_bytes=2**20)
        capped = production.instance_plan(pid, profile="large", total_memory_bytes=10 * 2**20)
        assert capped["memory_budget_bytes"] == 2**20 and capped["dimensions"] == plan["dimensions"]
    with pytest.raises(ValueError, match="memory envelope"):
        production.instance_plan(6, memory_bytes=1024)


def test_suite_manifest_sources_and_native_config_are_separate_from_verified():
    from scripts.build_production_tasks import build

    assert build(check=True)["references"] == 41
    manifest = suites.load_manifest("production41")
    assert Counter(row["track"] for row in manifest["problems"]) == {"design": 12, "numerical": 6, "implementation": 12, "backward": 11}
    assert len(select_problems(5, suite="production41")) == 41
    assert sum(len(select_problems(level)) for level in (1, 2, 3)) == 250
    config = parse_config(["--suite", "production41", "--level", "5", "--task-profile", "small", "--task-memory-mb", "64"])
    assert config.task_profile == "small" and config.task_memory_mb == 64
    with pytest.raises(ValueError, match="Level 5 requires"):
        RunConfig(level=5).validate()


def test_real_reference_prompt_and_payload_preserve_contract_without_generators(tmp_path):
    config = RunConfig(level=5, suite="production41", run_name="production", runs_dir=str(tmp_path), task_profile="small").validate()
    config.run_dir.mkdir()
    path = select_problems(5, problem_ids=[21])[21]
    source = path.read_text()
    Model, initial, inputs = load_original_model_and_inputs(source, {})
    with production.production_context(profile="small"):
        args = inputs()
        assert args[0].dtype == torch.float8_e4m3fn and args[2].dtype == torch.float32
        assert Model(*initial())(*args).dtype == torch.bfloat16
    prompt = generation.prompt_source(source, 5, 21)
    assert "def fp8_scaled_gemm" in prompt and "get_inputs" not in prompt and "make_inputs" not in prompt
    assert "float8_e4m3fn" in prompt and "scale" in prompt
    (config.run_dir / kernel_name(5, 21)).write_text(source.replace("class Model(", "class ModelNew("))
    payload = engine.task_payload(config, 21)
    assert payload["eval_config"]["precision"] == "bf16"
    assert payload["eval_config"]["production"]["task_sha256"] == fingerprint(task(21))
    old = fingerprint(payload)
    config.task_memory_mb //= 2
    assert fingerprint(engine.task_payload(config, 21)) != old


def test_all_generated_private_modules_have_real_cases():
    for pid in range(1, 42):
        path = ROOT / f"hidden_tests/level5/{pid}_hidden.py"
        module_spec = importlib.util.spec_from_file_location(f"production_hidden_{pid}", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cases = module.get_hidden_inputs()
        expected_count = 4 if task(pid)["family"] in {"normalization", "rmsnorm"} else 3
        assert len(cases) == expected_count and all(isinstance(row, list) and row for row in cases)
        if expected_count == 4:
            assert all(torch.isfinite(value).all() for value in tensors(production.execute(pid, *cases[-1])))


def test_production_policy_restores_context_and_rejects_changed_contract(monkeypatch):
    options = {"protocol": production.PROTOCOL, "task_sha256": fingerprint(task(3)), "profile": "small", "memory_bytes": 2**20}
    payload = {"level": 5, "problem_id": 3, "eval_config": {"production": options, "precision": "bf16"}}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: type("Caps", (), {"total_memory": 10 * 2**20})())
    @production.task_input_policy
    def evaluate(payload):
        args = production.make_inputs(3)
        assert args[0].dtype == torch.bfloat16 and args[1].dtype == torch.float32
        assert _process_input_tensor(args[1], "cpu", torch.bfloat16).dtype == torch.float32
        return {"correctness": True}
    result = evaluate(payload)
    assert result["metadata"]["production_instance"]["precision_family"] == "bf16"
    assert production.SETTINGS.get() == {}
    corrupted = copy.deepcopy(payload)
    corrupted["eval_config"]["production"]["task_sha256"] = "bad"
    with pytest.raises(ValueError, match="contract changed"):
        evaluate(corrupted)


@pytest.mark.parametrize("pid", range(13, 19))
def test_numerical_budget_runs_in_shared_correctness_comparator(pid, monkeypatch):
    from src.validation import compare_outputs

    options = {"protocol": production.PROTOCOL, "task_sha256": fingerprint(task(pid)), "profile": "small", "memory_bytes": 2**20}
    payload = {"level": 5, "problem_id": pid, "eval_config": {"production": options, "precision": "fp32"}}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: type("Caps", (), {"total_memory": 10 * 2**20})())
    @production.task_input_policy
    def evaluate(payload):
        expected = production.execute(pid, *production.make_inputs(pid, seed=12))
        assert compare_outputs(expected, expected.clone(), 1e-3) is None
        assert compare_outputs(expected, torch.zeros_like(expected), 1e-3) is not None
        # A tighter task budget overrides a permissive native tolerance.
        assert compare_outputs(torch.ones(10), torch.ones(10) * 1.1, 0.5) is not None
        return {"correctness": True}
    assert evaluate(payload)["metadata"]["production_instance"]["precision_contract"]
    assert compare_outputs(torch.ones(10), torch.ones(10) * 1.1, 0.5) is None


@pytest.mark.parametrize("pid", range(1, 42))
def test_zero_candidate_cannot_exploit_absolute_tolerance_in_any_production_family(pid, monkeypatch):
    from src.validation import compare_outputs

    spec = task(pid)
    options = {"protocol": production.PROTOCOL, "task_sha256": fingerprint(spec), "profile": "small", "memory_bytes": 2**20}
    payload = {"level": 5, "problem_id": pid, "eval_config": {"production": options,
               "precision": spec["fixed_activation_dtype"] or "fp32"}}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: type("Caps", (), {"total_memory": 10 * 2**20})())
    @production.task_input_policy
    def evaluate(payload):
        expected = production.execute(pid, *production.make_inputs(pid, seed=12))
        actual = tuple(torch.zeros_like(x) for x in expected) if isinstance(expected, tuple) else torch.zeros_like(expected)
        assert compare_outputs(expected, actual, 0.5) is not None
        return {"correctness": False}
    assert evaluate(payload)["correctness"] is False


@pytest.mark.parametrize("pid,method,accepted", [(13, "bf16", False), (13, "tf32", False),
                                                (13, "six_pass_bf16", True), (14, "bf16", True)])
def test_precision_design_distinguishes_emulation_and_different_error_budgets(pid, method, accepted, monkeypatch):
    from src.validation import compare_outputs

    options = {"protocol": production.PROTOCOL, "task_sha256": fingerprint(task(pid)), "profile": "small", "memory_bytes": 2**20}
    payload = {"level": 5, "problem_id": pid, "eval_config": {"production": options, "precision": "fp32"}}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: type("Caps", (), {"total_memory": 10 * 2**20})())
    @production.task_input_policy
    def evaluate(payload):
        x, weight = production.make_inputs(pid, seed=19)
        expected = production.execute(pid, x, weight)
        low_x, low_weight = x.bfloat16().float(), weight.bfloat16().float()
        if method == "tf32":
            def tf32_round(value):
                bits = value.contiguous().view(torch.int32)
                rounded = bits + 0xFFF + ((bits >> 13) & 1)
                return (rounded & -8192).view(torch.float32)
            actual = tf32_round(x) @ tf32_round(weight)
        elif method == "six_pass_bf16":
            rx = (x - low_x).bfloat16().float()
            rw = (weight - low_weight).bfloat16().float()
            tail_x = (x - low_x - rx).bfloat16().float()
            tail_w = (weight - low_weight - rw).bfloat16().float()
            actual = (low_x @ low_weight + rx @ low_weight + low_x @ rw + rx @ rw
                      + tail_x @ low_weight + low_x @ tail_w)
        else:
            actual = low_x @ low_weight
        assert (compare_outputs(expected, actual, 1e-3) is None) is accepted
        return {"correctness": accepted}
    evaluate(payload)


def test_native_generation_baseline_evaluation_and_scoring_keep_production_slots(tmp_path, monkeypatch):
    from kb_workflows.common import failure
    from kb_workflows.paper_analysis import analyze_paper

    config = RunConfig(level=5, suite="production41", run_name="native-production", runs_dir=str(tmp_path),
                       start=21, end=21, num_samples=2, task_profile="small", baseline_suite=False,
                       hardware="test", gpu_arch="Ampere", generation="local").validate()
    observed = []
    def evaluate_tasks(config, payloads):
        for key, payload in payloads:
            observed.append(payload)
            assert payload["level"] == 5 and payload["eval_config"]["production"]["profile"] == "small"
            assert payload["eval_config"]["precision"] == "bf16"
            yield key, failure("Deliberately missing hardware in protocol regression")
    monkeypatch.setattr(engine, "evaluate_tasks", evaluate_tasks)
    hidden = tmp_path / "hidden_tests/level5/21_hidden.py"
    hidden.parent.mkdir(parents=True)
    hidden.write_text((ROOT / "hidden_tests/level5/21_hidden.py").read_text())
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    directory = generation.prepare_job(config, 21, 0)
    child = RunConfig(**read_json(directory / "job_config.json")).validate()
    assert child.suite == "production41" and child.level == 5
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for sid in range(2):
        (config.run_dir / kernel_name(5, 21, sid)).write_text("candidate placeholder; never executed")
    engine.record_baseline(config)
    engine.evaluate_run(config)
    report = analyze_paper(config)
    assert len(observed) == 5  # baseline + 2 standard + 2 hidden.
    assert report["problem_ids"] == [21] and len(report["details"]) == 2
    assert report["score_macro"] == 0
    assert read_json(config.run_dir / "validation_summary.json")["counts"]["declared_samples"] == 2


def test_harbor_production_child_keeps_suite_and_public_math_only(tmp_path):
    from kb_workflows.harbor import export_tasks

    config = RunConfig(level=5, suite="production41", run_name="harbor-production", runs_dir=str(tmp_path),
                       start=31, end=31, task_profile="small", hardware="test", gpu_arch="Ampere").validate()
    target = export_tasks(config) / "l5-p31-s0"
    child = RunConfig(**read_json(target / "tests/config.json")["config"]).validate()
    assert child.suite == "production41" and child.task_profile == "small"
    public = (target / "environment/model.py").read_text()
    assert "def attention" in public and "def vjp" in public
    assert "make_inputs" not in public and "get_hidden_inputs" not in public


@pytest.mark.parametrize("track,first,last", [("design", 1, 12), ("numerical", 13, 18),
                                               ("implementation", 19, 30), ("backward", 31, 41)])
def test_each_track_is_a_separate_declared_selection(track, first, last):
    config = parse_config(["--suite", "production41", "--level", "5", "--task-track", track])
    selected = select_problems(5, suite=config.suite, track=config.task_track)
    assert list(selected) == list(range(first, last + 1))
    assert suites.selection_receipt(config)["track"] == track


def test_track_reports_separate_backward_and_do_not_rank_the_mixed_collection(tmp_path):
    from kb_workflows.paper_analysis import analyze_paper

    config = RunConfig(level=5, suite="production41", run_name="separate-tracks", runs_dir=str(tmp_path),
                       hardware="test", gpu_arch="Ampere", task_profile="small").validate()
    report = analyze_paper(config)
    assert set(report["score_by_track"]) == {"design", "numerical", "implementation", "backward"}
    assert not report["leaderboard_eligible"]
    config.task_track = "backward"
    report = analyze_paper(config)
    assert report["problem_ids"] == list(range(31, 42)) and set(report["score_by_track"]) == {"backward"}
    assert report["leaderboard_eligible"]

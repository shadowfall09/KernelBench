"""Device contract regressions only; SDK execution is replaced with explicit fakes."""

import copy
import inspect
import json
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from kb_workflows import engine
from kb_workflows.analysis import paper_analysis
from kb_workflows.common import ROOT, fingerprint, kernel_name, select_problems
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.evaluation.baseline_variants import variants_for
from kb_workflows.hardware import devices, neuron_device, neuron_worker
from kb_workflows.hardware import neuron_tables as neuron_profile


def neuron_config(tmp_path, **options):
    return RunConfig(device_backend="neuron", hardware="Trainium2", generation="local", level=5, suite="production41",
        start=19, end=19, task_profile="small", run_name="neuron-test", runs_dir=str(tmp_path), num_correct_trials=1,
        num_perf_trials=3, paired_trials=10, final_repeats=2, enable_ncu=False, **options).validate()


def test_device_config_is_lightweight_and_preserves_existing_gpu_configs(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[device]\nbackend="neuron"\nhardware="Trainium2"\ndevices=[0,2]\npython="vendor/bin/python"\n'
                    '[run]\ngeneration="local"\nrun_name="native_neuron"\n')
    config = parse_config(["--config", str(path), "--dry-run"])
    assert (config.device_backend, config.hardware, config.gpu_arch, config.gpus) == ("neuron", "Trainium2", "trn2", "0,2")
    assert config.device_python == str(tmp_path / "vendor/bin/python")
    assert variants_for(config, 1) == ("neuron_eager", "neuron_compile")
    old = parse_config(["--gpu-profile", "H100", "--dry-run"])
    assert old.device_backend == "cuda" and old.gpu_arch == "Hopper"
    path.write_text(path.read_text() + '\n[gpu]\nprofile="H100"\n')
    with pytest.raises(ValueError, match="either"):
        parse_config(["--config", str(path)])


@pytest.mark.parametrize("section", [
    {"backend": "xla"}, {"backend": "neuron", "hardware": "Trainium1"},
    {"backend": "neuron", "hardware": "Trainium2", "target": "trn3"},
    {"backend": "neuron", "hardware": "Trainium2", "devices": [0, 0]},
    {"backend": "cuda", "python": ""}, {"backend": "cuda", "unknown": 1},
])
def test_device_configuration_rejects_ambiguous_targets(section):
    with pytest.raises(ValueError):
        devices.device_options(section)


@pytest.mark.parametrize("options", [{"evaluation": "docker"}, {"evaluation": "modal"}, {"generation": "harbor"},
                                      {"baseline_portfolio": "extended"}, {"compile_backend": "inductor"}, {"enable_ncu": True}])
def test_neuron_cannot_use_cuda_images_compilers_or_cloud_defaults(tmp_path, options):
    base = dict(device_backend="neuron", hardware="Trainium2", generation="local", runs_dir=str(tmp_path))
    with pytest.raises(ValueError):
        RunConfig(**{**base, **options}).validate()


def test_visible_neuron_core_mapping_does_not_use_cuda_allocation(monkeypatch):
    from kb_workflows.runtime.transports import resolve_devices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "2-4,7")
    assert resolve_devices("0,3", backend="neuron") == ["2", "7"]
    with pytest.raises(ValueError, match="outside"):
        resolve_devices("4", backend="neuron")
    for bad in ("", "1,1", "4-2", "*", "0-10000"):
        with pytest.raises(ValueError):
            neuron_device.core_ids(bad)


def test_neuron_identity_uses_device_serial_core_loaded_driver_and_sdk(tmp_path):
    info = tmp_path / "sysfs/neuron3/info"
    (info / "architecture").mkdir(parents=True)
    (info / "serial_number").write_text("SERIAL-unique")
    (info / "architecture/device_name").write_text("Trainium2")
    driver = tmp_path / "loaded_driver"
    driver.write_text("2.33")
    row = {"neuron_device": 3, "neuroncore_ids": [12, 13], "memory_size": 96 * 1024**3, "bdf": "cc:00.0"}
    value = neuron_device.inventory_record([row], core=12, sysfs=tmp_path / "sysfs", driver=driver)
    assert value["serial_number"] == "SERIAL-unique" and value["physical_core"] == 12 and value["driver_version"] == "2.33"
    with pytest.raises(ValueError, match="absent or ambiguous"):
        neuron_device.inventory_record([row, row], core=12, sysfs=tmp_path / "sysfs", driver=driver)
    neuron_device.native_versions("2.14.0+cpu", "2.14.3.0.4527", "0.6.0")
    for versions in (("2.12.0", "2.14.3.0.1", "0.6.0"), ("2.14.0", "2.9.0.2.15", "0.6.0"), ("2.14.0", "2.14.3.0.1", "0.3.0")):
        with pytest.raises(ValueError, match="native torch"):
            neuron_device.native_versions(*versions)


def test_two_adapters_expose_the_five_protocol_methods_without_sdk_imports():
    script = ('import sys; from kb_workflows.hardware import devices, neuron_device, neuron_tables as neuron_profile; assert not {"torch", "torch_neuronx", "nki"} & sys.modules.keys(); from kb_workflows.analysis.calibration import CUDADevice; assert isinstance(CUDADevice.__new__(CUDADevice), devices.DeviceAdapter); assert isinstance(neuron_device.NeuronDevice.__new__(neuron_device.NeuronDevice), devices.DeviceAdapter)')
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_neuron_instruction_union_does_not_sum_overlaps_or_count_instructions_as_kernels():
    trace = {"instruction": [{"timestamp": 0, "duration": 100, "label": "TensorMatrix"},
                              {"timestamp": 50, "duration": 100, "label": "Vector"},
                              {"timestamp": 120, "duration": 80, "label": "DMA"}], "runtime": [{"duration": 999999}]}
    value = neuron_profile.timeline(trace, timestamp_unit="us", window=[0, 250], origins={"0": "custom"})
    assert value["device_busy_ms"] == pytest.approx(0.2)
    assert value["per_engine_busy_ms"]["tensor"] == pytest.approx(0.1)
    assert value["custom_exclusive_fraction"] == pytest.approx(0.25)
    assert value["wall_ms"] is None and value["kernel_count"] is None and value["instruction_count"] == 3
    assert value["unknown_origin_count"] == 2 and not value["hardware_counters_collected"]
    for options in ({"timestamp_unit": "auto", "window": [0, 250]}, {"timestamp_unit": "ns", "window": [250, 0]}):
        with pytest.raises(ValueError):
            neuron_profile.timeline(trace, **options)
    with pytest.raises(ValueError, match="host/runtime"):
        neuron_profile.timeline({"runtime": [{}]}, timestamp_unit="ns", window=[0, 10])


class SimulatedNeuron:
    simulated, device = True, torch.device("cpu")

    def caps(self):
        return {"device_backend": "neuron", "name": "SIMULATED", "available": True, "uuid": "simulated-neuron-core",
                "capability": ["NKI", "LNC1"], "total_memory": 10**11, "multiprocessors": 1, "l2_bytes": None,
                "torch_version": "test", "cuda_version": None, "hip_version": None, "driver_version": "test"}

    def synchronize(self):
        pass


@pytest.fixture
def payload(tmp_path, monkeypatch):
    config = neuron_config(tmp_path)
    config.run_dir.mkdir()
    path = select_problems(5, problem_ids=[19])[19]
    (config.run_dir / kernel_name(5, 19)).write_text(path.read_text().replace("class Model(", "class ModelNew("))
    worker_root = tmp_path / "worker"
    hidden = worker_root / "hidden_tests/level5/19_hidden.py"
    hidden.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "hidden_tests/level5/19_hidden.py", hidden)
    monkeypatch.setattr(neuron_worker, "ROOT", worker_root)
    return config, engine.task_payload(config, 19)


@pytest.mark.parametrize("pid", [pid for pid in range(1, 42) if pid != 21])
def test_production_neuron_frontend_preserves_contracts_for_each_supported_storage_task(pid, tmp_path, monkeypatch):
    options = neuron_config(tmp_path).to_dict()
    options.update(start=pid, end=pid, num_perf_trials=1)
    config = RunConfig(**options).validate()
    config.run_dir.mkdir()
    path = select_problems(5, problem_ids=[pid])[pid]
    (config.run_dir / kernel_name(5, pid)).write_text(path.read_text().replace("class Model(", "class ModelNew("))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(neuron_worker, "ROOT", tmp_path / "worker")
    try:
        value = neuron_worker.evaluate(engine.task_payload(config, pid), adapter=SimulatedNeuron())
    finally:
        torch.set_num_threads(threads)
    assert value["correctness"] and value["metadata"]["simulated_execution"]
    assert value["metadata"]["production_instance"]["problem_id"] == pid


def test_native_neuron_worker_runs_reference_contracts_and_varied_checks_on_simulated_device(payload, monkeypatch):
    config, request = payload
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    result = neuron_worker.evaluate(request, adapter=SimulatedNeuron())
    assert result["correctness"] and result["metadata"]["simulated_execution"]
    assert result["metadata"]["correctness_scope"].startswith("paired_orders_and_CPU")
    assert len(result["runtime_stats"]["samples_ms"]) == config.num_perf_trials
    assert result["metadata"]["timing_receipt"]["timed_outputs_verified"]
    assert paper_analysis.measurement_evidence(result, candidate=True) == "simulated_device_execution_cannot_qualify"
    baseline = engine.task_payload(config, 19, task="baseline")
    baseline["eval_config"]["baseline_variant"] = "neuron_compile"
    value = neuron_worker.evaluate(baseline, adapter=SimulatedNeuron())
    assert value["baseline_correctness"] and value["baseline_variant"] == "neuron_compile"
    unmarked = SimulatedNeuron()
    unmarked.simulated = False
    assert neuron_worker.evaluate(request, adapter=unmarked)["metadata"]["simulated_execution"]


def test_neuron_worker_uses_frozen_private_seeds_and_rejects_changed_hidden_source(payload):
    config, _ = payload
    hidden = engine.task_payload(config, 19, hidden=True)
    hidden["eval_config"]["hidden_seeds"] = [991, 992]
    result = neuron_worker.evaluate(hidden, adapter=SimulatedNeuron())
    assert result["correctness"] and result["metadata"]["hidden_completed_checks"] == 6
    labels = [row["label"] for row in result["metadata"]["correctness_cases"]]
    assert all(label.startswith(("hidden/991/", "hidden/992/")) for label in labels)
    assert any(row["input_signature"][0]["stride"][-1] == 2 for row in result["metadata"]["correctness_cases"])
    hidden["eval_config"]["hidden_source_sha256"] = "modified"
    with pytest.raises(ValueError, match="hidden test sources"):
        neuron_worker.evaluate(hidden, adapter=SimulatedNeuron())


def test_neuron_worker_rejects_wrong_outputs_and_changed_precision(payload):
    _, request = payload
    wrong = copy.deepcopy(request)
    wrong["kernel_src"] = 'import torch\nclass ModelNew(torch.nn.Module):\n def forward(self,x,w): return torch.zeros(x.shape[0],w.shape[0],dtype=x.dtype)\n'
    with pytest.raises(ValueError, match="mismatch|contract|budget"):
        neuron_worker.evaluate(wrong, adapter=SimulatedNeuron())
    invalid = copy.deepcopy(request)
    # The public task's fixed BF16 output cannot be silently promoted to FP32.
    invalid["kernel_src"] += '\n_old=ModelNew.forward\nModelNew.forward=lambda self,*x:_old(self,*x).float()\n'
    with pytest.raises(ValueError, match="dtype|mismatch"):
        neuron_worker.evaluate(invalid, adapter=SimulatedNeuron())


def test_neuron_worker_paired_timing_reuses_native_balance_and_checks_every_output(payload, monkeypatch):
    config, _ = payload
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    request = engine.paired_payload(config, 19, 0, variant="neuron_compile")
    result = neuron_worker.evaluate(request, adapter=SimulatedNeuron())
    paired = result["metadata"]["paired_timing"]
    assert paired["all_outputs_verified"] and paired["num_pairs"] == 10
    assert paired["baseline_variant"] == "neuron_compile"
    assert all(sorted(row["order"]) == ["B", "C", "R"] for row in paired["pairs"])


def test_neuron_interpreter_is_local_only_and_device_changes_invalidate_evaluation(payload, tmp_path):
    config, _ = payload
    before = fingerprint(engine.task_payload(config, 19))
    config.device_python = str(tmp_path / "neuron-venv/bin/python")
    changed = engine.task_payload(config, 19)
    assert changed["worker_python"] == config.device_python and fingerprint(changed) != before
    config.evaluation = "queue"
    assert "worker_python" not in engine.task_payload(config, 19)


def test_local_transport_pins_neuron_cores_and_vendor_interpreter_before_torch_import(payload, tmp_path, monkeypatch):
    from kb_workflows.runtime import transports

    _, request = payload
    request["worker_python"] = "/vendor/bin/python"
    observed = []
    class Process:
        returncode = 0
        def __init__(self, command, **options):
            observed.append((command, options))
            Path(command[-1]).write_text(json.dumps({"compiled": True, "correctness": False}))
        def wait(self, **kwargs):
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(transports.subprocess, "Popen", Process)
    monkeypatch.setattr(transports, "terminate", lambda process: None)
    result = transports.local_evaluate(request, device="7", log_path=tmp_path / "worker.log")
    command, options = observed[0]
    assert result["compiled"] and command[0] == "/vendor/bin/python"
    assert options["env"]["NEURON_RT_VISIBLE_CORES"] == "7" and options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert options["env"]["TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION"] == "1"
    assert not Path(command[-2]).exists() and not Path(command[-1]).exists()


def test_native_clock_waits_for_work_and_includes_consumption_without_running_device_code():
    instance = neuron_device.NeuronDevice.__new__(neuron_device.NeuronDevice)
    instance.device, instance.torch = torch.device("cpu"), torch
    calls = []
    instance._sync = lambda device: calls.append("sync")
    times = iter(range(16))
    instance._clock = lambda: next(times) / 1000
    measured = instance.time(lambda: torch.ones(2), trials=5, warmup=3)
    assert measured == pytest.approx([1] * 5) and calls == ["sync"] * 16
    with pytest.raises(ValueError, match="explicit scratch"):
        instance.flush()


def test_neuron_calibration_uses_native_compile_and_synchronized_wall_schema_with_fake_probes(monkeypatch):
    from kb_workflows.analysis import calibration
    from kb_workflows.evaluation import measurement

    class Tensor:
        def to(self, device):
            return self
        def copy_(self, source):
            return self
        def add_(self, value):
            return self
    calls = []
    stub = SimpleNamespace(no_grad=nullcontext, uint8="uint8", float32="fp32", float16="fp16", bfloat16="bf16",
        ones=lambda *a, **k: Tensor(), zeros=lambda *a, **k: Tensor(), empty_like=lambda x: Tensor(),
        randn=lambda *a, **k: Tensor())
    def compile_(model, **options):
        calls.append(options)
        return lambda *a: Tensor()
    stub.compile = compile_
    instance = neuron_device.NeuronDevice.__new__(neuron_device.NeuronDevice)
    instance.device, instance.torch = "neuron:0", stub
    instance.caps = SimulatedNeuron().caps
    instance.time = lambda *a, **k: [1, 2, 3, 4, 5]
    instance._sync = lambda device: None
    clock = iter(range(100))
    instance._clock = lambda: next(clock) / 1000
    monkeypatch.setattr(measurement, "consume_output", lambda output: None)
    result = instance.calibrate(matrix_sizes=(128, 256), memory_bytes=1024, trials=5)
    assert len(calls) == 6 and all(row == {"backend": "neuron", "fullgraph": True, "dynamic": False} for row in calls)
    assert "fp64_flops" not in result["probes"] and set(result["probes"]) == {"memory_bytes", "fp32_flops", "fp16_flops", "bf16_flops"}
    assert result["probes"]["memory_bytes"]["rate_policy"] == "work_over_median_synchronized_wall_time"
    assert calibration.validate(result, SimulatedNeuron().caps())["receipt_sha256"] == result["receipt_sha256"]


@pytest.mark.parametrize("pid", [1, 3, 27, 30])
def test_neuron_library_baselines_preserve_production_semantics(pid, monkeypatch):
    from kb_workflows import production

    monkeypatch.setattr(torch, "compile", lambda model, **options: model)
    args = production.make_inputs(pid, profile="small", seed=8)
    expected = production.execute(pid, *args)
    library = neuron_worker.prepare_variant(None, "neuron_library", level=5, pid=pid)
    from src.validation import compare_outputs
    assert compare_outputs(expected, library(*args), 1e-5) is None


def test_nki_compiler_can_inspect_frozen_source_and_changed_file_is_rejected(payload):
    config, request = payload
    value = neuron_worker.evaluate(request, adapter=SimulatedNeuron())
    path = neuron_worker.ROOT / "cache/workflow" / fingerprint(request) / "candidate.py"
    namespace = neuron_worker.load_source(request["kernel_src"], "inspection_test", filename=path)
    assert "class ModelNew" in inspect.getsource(namespace["ModelNew"])
    assert value["metadata"]["compiler_source_sha256"]["candidate.py"]
    path.write_text("# changed compiled source")
    with pytest.raises(ValueError, match="compiler source file"):
        neuron_worker.evaluate(engine.task_payload(config, 19), adapter=SimulatedNeuron())


def test_neuron_public_development_never_accepts_fp8_storage_downgrade(tmp_path):
    options = neuron_config(tmp_path).to_dict()
    options.update(start=21, end=21)
    config = RunConfig(**options).validate()
    config.run_dir.mkdir()
    source = select_problems(5, problem_ids=[21])[21].read_text()
    (config.run_dir / kernel_name(5, 21)).write_text(source.replace("class Model(", "class ModelNew("))
    with pytest.raises(ValueError, match="float8 storage bridge"):
        neuron_worker.evaluate(engine.task_payload(config, 21), adapter=SimulatedNeuron())


def test_native_caps_bind_compiler_versions_and_physical_core_without_sdk_execution(monkeypatch):
    instance = neuron_device.NeuronDevice.__new__(neuron_device.NeuronDevice)
    instance.device = SimpleNamespace(index=0)
    instance.torch = SimpleNamespace(__version__="2.14.0")
    instance.backend = SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(total_memory=10**10))
    instance._versions = {"torch_neuronx": "2.14.3.0.1", "nki": "0.6.0", "neuronx_cc": "test"}
    physical = {"name": "Trainium2", "total_memory": 10**10, "driver_version": "2.33", "serial_number": "simulated", "physical_core": 7}
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "7")
    monkeypatch.setattr(neuron_device.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="[]"))
    monkeypatch.setattr(neuron_device, "inventory_record", lambda rows, *, core: {**physical, "physical_core": core})
    first = instance.caps()
    instance._versions = {**instance._versions, "nki": "0.7.0"}
    second = instance.caps()
    assert first["uuid"] == second["uuid"] and first["driver_version"] != second["driver_version"]
    assert first["physical_device"]["physical_core"] == 7 and not first["float64_arithmetic"]
    assert first["cuda_version"] is None and not first["cpu_fallback_allowed"]


def test_neuron_profile_api_retains_native_artifact_hashes_with_a_simulated_sdk(tmp_path, monkeypatch):
    directory, observed, events = tmp_path / "profile", [], []
    monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_DIR", str(tmp_path / "source_cache"))
    class Capture:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            observed[-1]["on_trace_ready"](self)
            events.append("capture_closed")
        def export_chrome_trace(self, path):
            Path(path).write_text('{"traceEvents":[]}')
    class Exporter:
        def __init__(self, config):
            self.config = config
        def export_trace(self, capture):
            (directory / "fixture.neff").write_bytes(b"SIMULATED NEFF")
            (directory / "fixture.ntff").write_bytes(b"SIMULATED NTFF")
    vendor, profiling = ModuleType("torch_neuronx"), ModuleType("torch_neuronx.profiling")
    vendor.__path__ = []
    profiling.NeuronConfig = lambda **options: options
    profiling.NeuronProfiler = Exporter
    profiling.ProfileMode = SimpleNamespace(DEVICE="device", RUNTIME="runtime")
    monkeypatch.setitem(sys.modules, "torch_neuronx", vendor)
    monkeypatch.setitem(sys.modules, "torch_neuronx.profiling", profiling)
    def profile(**options):
        observed.append(options)
        return Capture()
    stub = SimpleNamespace(profiler=SimpleNamespace(profile=profile, record_function=lambda *a: nullcontext(),
                                                    ProfilerActivity=SimpleNamespace(CPU="cpu", PrivateUse1="neuron")))
    instance = neuron_device.NeuronDevice.__new__(neuron_device.NeuronDevice)
    instance.device, instance.torch, instance._sync = "neuron:0", stub, lambda device: None
    def verify(output):
        assert events == ["capture_closed"] and output == "profiled_output"
        events.append("verified")
    receipt = instance.profile(lambda: "profiled_output", directory=directory, verifier=verify)
    assert set(receipt["artifact_files"]) == {"framework_trace.json", "fixture.neff", "fixture.ntff"}
    assert receipt["artifact_file_bytes"] == {name: (directory / name).stat().st_size for name in receipt["artifact_files"]}
    assert observed[0]["experimental_config"]["modes"] == ["device", "runtime"]
    assert observed[0]["experimental_config"]["neff_cache_dir"] == str(tmp_path / "source_cache")
    assert receipt["output_verified"] and events == ["capture_closed", "verified"]
    assert not receipt["hardware_counters_collected"] and receipt["timings_are_diagnostic"]
    with pytest.raises(ValueError, match="fresh artifact"):
        instance.profile(lambda: None, directory=directory)


class ProfiledSimulatedNeuron(SimulatedNeuron):
    def __init__(self, *, corrupt=False):
        self.corrupt = corrupt

    def profile(self, function, *, directory, verifier):
        assert not torch.is_grad_enabled()
        output = function()
        if self.corrupt:
            output = neuron_worker.map_tensors(output, lambda value: value + 1)
        verifier(output)
        return {"output_verified": True, "timings_are_diagnostic": True}


@pytest.mark.parametrize("corrupt", [False, True])
def test_neuron_profile_checks_the_captured_output_and_restores_cache(payload, monkeypatch, corrupt):
    _, request = payload
    request["task"] = "profile"
    monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_DIR", "previous_cache")
    import os

    if corrupt:
        with pytest.raises(ValueError, match="mismatch|contract"):
            neuron_worker.evaluate(request, adapter=ProfiledSimulatedNeuron(corrupt=True))
    else:
        result = neuron_worker.evaluate(request, adapter=ProfiledSimulatedNeuron())
        assert result["output_verified"] and result["metadata"]["profile_call"]["output_verified"]
        assert result["metadata"]["profile_call"]["no_grad"]
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == "previous_cache"


def test_neuron_profile_uses_pre_forward_state_and_matching_rng(payload, monkeypatch):
    from kb_workflows.evaluation import evaluate_task

    _, request = payload
    source = '''import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.calls=0
 def forward(self,x):
  self.calls+=1
  assert not torch.is_grad_enabled()
  return x+self.calls+torch.rand_like(x)
def get_init_inputs(): return []
def get_inputs(): return [torch.randn(2,3)]
'''
    request.update(task="profile", level=1, problem_id=1, ref_arch_src=source,
                   kernel_src=source.replace("class Model(", "class ModelNew("))
    request["eval_config"].update(num_correct_trials=3, precision="fp32")
    request["eval_config"].pop("production", None)
    monkeypatch.setattr(evaluate_task, "validate_sources", lambda payload: payload["ref_arch_src"])
    result = neuron_worker.evaluate(request, adapter=ProfiledSimulatedNeuron())
    assert result["output_verified"]
    assert result["metadata"]["profile_call"]["state_scope"] == "continuous_state; pre_forward_reference_snapshot"
    assert not result["metadata"]["profile_call"]["independent_oracle_ticket"]


def test_neuron_profile_isolates_inputs_before_the_captured_interval(payload, monkeypatch):
    from kb_workflows.evaluation import evaluate_task

    _, request = payload
    source = '''import torch
class Model(torch.nn.Module):
 def forward(self,x): return x+1
def get_init_inputs(): return []
def get_inputs(): return [torch.randn(2,3)]
'''
    request.update(task="profile", level=1, problem_id=1, ref_arch_src=source,
                   kernel_src=source.replace("class Model(", "class ModelNew("))
    request["eval_config"].update(precision="fp32")
    request["eval_config"].pop("production", None)
    monkeypatch.setattr(evaluate_task, "validate_sources", lambda payload: payload["ref_arch_src"])
    active, captured_clones = False, []
    original = torch.Tensor.clone

    def clone(tensor, *args, **kwargs):
        if active:
            captured_clones.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    class IntervalAdapter(ProfiledSimulatedNeuron):
        def profile(self, function, *, directory, verifier):
            def captured():
                nonlocal active
                active = True
                try:
                    return function()
                finally:
                    active = False
            return super().profile(captured, directory=directory, verifier=verifier)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    result = neuron_worker.evaluate(request, adapter=IntervalAdapter())
    assert captured_clones == [] and result["output_verified"]
    capture = result["capture_receipt"]
    assert capture["simulated"] and capture["binding"]["input_probe_sha256"]
    assert capture["compiler_source_sha256"] == result["metadata"]["compiler_source_sha256"]


@pytest.mark.parametrize("throws", [False, True])
def test_environment_guard_restores_grad_mode_even_after_candidate_failure(throws):
    from kb_workflows.evaluation.strict_validation import environment_guard

    previous = torch.is_grad_enabled()
    changes = []
    with torch.no_grad():
        try:
            with environment_guard(changes):
                torch.set_grad_enabled(True)
                if throws:
                    raise ValueError("candidate failed")
        except ValueError:
            assert throws
        assert not torch.is_grad_enabled()
    assert changes == ["grad_enabled"] and torch.is_grad_enabled() == previous


@pytest.mark.parametrize("kind", ["environment", "source_file"])
def test_neuron_first_execution_cannot_mutate_environment_or_frozen_compiler_source(payload, kind):
    _, request = payload
    request["kernel_src"] += '\nimport os\nfrom pathlib import Path\n_original=ModelNew.forward\n'
    action = 'os.environ["KB_NEURON_TEST_SETTING"]="changed"' if kind == "environment" else 'Path(__file__).write_text("# corrupt compiled source")'
    request["kernel_src"] += f'\ndef replaced(self,*args):\n {action}\n return _original(self,*args)\nModelNew.forward=replaced\n'
    with pytest.raises(ValueError, match="process settings|source files changed"):
        neuron_worker.evaluate(request, adapter=SimulatedNeuron())


def test_neuron_auxiliary_oracles_can_run_on_cpu_without_downgrading_the_device_precision():
    from kb_workflows.evaluation.evidence import accepted
    from kb_workflows.evaluation.strict_validation import run_checks
    from kb_workflows.numerics.contract_specs import contract_for

    source = select_problems(1, problem_ids=[2])[2].read_text()
    contract = contract_for(1, 2, source)
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a @ b
    checks = run_checks(Model(), Model(), contract, hidden=True, seed=7, device="cpu", dtype=torch.float32,
                        set_seed=torch.manual_seed, oracle_on_cpu=True)
    assert all(accepted(item) for item in checks.values())
    assert checks["precision_contract"]["values"][0]["oracle_dtype"] == "float64/complex128"


def test_neuron_reliability_advances_stateful_models_equally_without_extra_candidate_calls(payload, monkeypatch):
    from kb_workflows.evaluation import evaluate_task

    _, request = payload
    source = '''import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer("calls",torch.zeros(()))
 def forward(self,x):
  self.calls.add_(1)
  return x+self.calls
def get_init_inputs(): return []
def get_inputs(): return [torch.randn(2,3)]
'''
    request.update(level=1, problem_id=1, ref_arch_src=source, kernel_src=source.replace("class Model(", "class ModelNew("))
    request["eval_config"].update(num_correct_trials=3, precision="fp32")
    request["eval_config"].pop("production", None)
    monkeypatch.setattr(evaluate_task, "validate_sources", lambda payload: payload["ref_arch_src"])
    value = neuron_worker.evaluate(request, adapter=SimulatedNeuron())
    assert value["correctness"] and len(value["metadata"]["reliability_checks"]) == 3
    assert all(row["orders"] == ["candidate_first", "reference_first"] for row in value["metadata"]["reliability_checks"])


def test_varied_timing_replays_neuron_rng_for_the_oracle_then_restores_candidate_progress(monkeypatch):
    from kb_workflows.evaluation import measurement

    state, candidate_seeds, oracle_seeds = {"value": 0}, [], []
    def set_(value, device):
        state["value"] = value
    proxy = SimpleNamespace(no_grad=torch.no_grad, device=lambda name: SimpleNamespace(type="neuron"),
        get_rng_state=torch.get_rng_state, set_rng_state=torch.set_rng_state,
        neuron=SimpleNamespace(get_rng_state=lambda device: state["value"], set_rng_state=set_))
    monkeypatch.setattr(measurement, "torch", proxy)
    monkeypatch.setattr(measurement, "synchronize", lambda device: None)
    def model(x):
        candidate_seeds.append(state["value"])
        state["value"] += 10
        return x + 1
    def verifier(args, output):
        oracle_seeds.append(state["value"])
        state["value"] = 999
    values = measurement.synchronized_times(model, [torch.ones(2)], "neuron:0", num_warmup=1, num_trials=2, verifier=verifier)
    assert len(values) == 2 and candidate_seeds == oracle_seeds == [0, 10, 20]
    assert state["value"] == 30


def test_existing_harbor_verifier_exports_the_complete_shared_device_digest(tmp_path, monkeypatch):
    from kb_workflows import common
    from kb_workflows.runtime import harbor

    config = RunConfig(run_name="harbor-device-proof", runs_dir=str(tmp_path), hardware="H100", start=90, end=90,
                       generation="local", stage="export-harbor").validate()
    task = harbor.write_task(tmp_path / "task", config, 90)
    expected = common.evaluator_digest()
    monkeypatch.setattr(common, "ROOT", task / "tests/repo")
    assert common.evaluator_digest() == expected


def test_neuron_small_qualification_executes_reviewed_cases_even_for_current_level_plans(tmp_path, monkeypatch):
    options = neuron_config(tmp_path).to_dict()
    options.update(level=1, suite="full", start=2, end=2)
    config = RunConfig(**options).validate()
    config.run_dir.mkdir()
    source = select_problems(1, problem_ids=[2])[2].read_text()
    (config.run_dir / kernel_name(1, 2)).write_text(source.replace("class Model(", "class ModelNew(").replace("super(Model,", "super(ModelNew,"))
    monkeypatch.setattr(neuron_worker, "ROOT", tmp_path / "worker")
    result = neuron_worker.evaluate(engine.task_payload(config, 2, task="qualify_small"), adapter=SimulatedNeuron())
    assert result["correctness"] and result["runtime"] == -1
    assert result["metadata"]["input_signature"] == []
    assert result["metadata"]["evidence_receipt"]["checks"]["precision_contract"]["status"] == "passed"

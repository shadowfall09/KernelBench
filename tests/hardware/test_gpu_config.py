"""GPU config precedence, portable task settings and early configuration errors."""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from kb_workflows import cli, engine
from kb_workflows.agents import generation
from kb_workflows.common import ROOT, fingerprint
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.hardware import gpu
from kb_workflows.runtime import harbor
from kb_workflows.runtime.transports import resolve_devices


def write_config(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_profile_resolves_one_target_for_every_backend(tmp_path):
    path = write_config(tmp_path, '''
[gpu]
profile = "RTX_A6000"
devices = [0, 1]
[run]
generation = "harbor"
evaluation = "docker"
start = 90
end = 90
run_name = "configured"
num_samples = 2
''')
    config = parse_config(["--config", str(path)])
    assert (config.hardware, config.gpu_arch, config.gpus, config.gpu) == ("RTX_A6000", "Ampere", "0,1", "")
    assert (config.start, config.end, config.num_samples) == (90, 90, 2)
    config.require_gpu_target()
    prompt = generation.build_prompt(config, 90)
    assert "RTX_A6000" in prompt and "Ampere" in prompt
    settings = engine.settings(config)
    assert settings["hardware"] == "RTX_A6000" and settings["gpu_arch"] == "Ampere"


def test_cli_overrides_file_including_values_equal_to_defaults(tmp_path):
    path = write_config(tmp_path, '''
[gpu]
profile = "RTX_A6000"
devices = [2, 3]
[run]
parallel = 9
precision = "bf16"
resume = true
hidden_tests = false
''')
    config = parse_config(["--config", str(path), "--parallel", "4", "--precision", "fp32",
                           "--no-resume", "--hidden-tests", "--gpus", "0"])
    assert (config.parallel, config.precision, config.resume, config.hidden_tests, config.gpus) == (4, "fp32", False, True, "0")


@pytest.mark.parametrize("target_flags", [["--gpu-profile", "H100"], ["--hardware", "H100"], ["--gpu", "H100"]])
def test_cli_target_change_does_not_inherit_old_architecture(tmp_path, target_flags):
    path = write_config(tmp_path, '[gpu]\nprofile = "RTX_A6000"\ndevices = [1]\n')
    config = parse_config(["--config", str(path), *target_flags])
    assert (config.hardware, config.gpu_arch, config.gpu, config.gpus) == ("H100", "Hopper", "H100", "1")


def test_inline_gpu_needs_no_registry_edit(tmp_path, monkeypatch):
    path = write_config(tmp_path, '''
[gpu]
hardware = "My_GPU"
architecture = "8.6;8.6+PTX"
devices = [0, 1]
''')
    config = parse_config(["--config", str(path)])
    assert config.gpu_arch == "8.6,8.6+PTX"
    assert config.hardware == "My_GPU" and not config.gpu
    config.require_gpu_target()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,GPU-allocated")
    assert resolve_devices(config.gpus) == ["3", "GPU-allocated"]
    from src.utils import set_gpu_arch

    set_gpu_arch(config.gpu_arch.split(","))
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.6;8.6+PTX"


def test_gpu_presets_are_loaded_from_data(tmp_path, monkeypatch):
    profiles = tmp_path / "gpu_profiles.toml"
    profiles.write_text('[profiles.Future_GPU]\narchitecture = "10.0"\nmodal_type = "cloud-gpu"\n')
    monkeypatch.setattr(gpu, "PROFILES_PATH", profiles)
    config = parse_config(["--gpu-profile", "Future_GPU"])
    assert (config.hardware, config.gpu_arch, config.gpu) == ("Future_GPU", "10.0", "cloud-gpu")


def test_config_relative_paths_follow_file_not_working_directory(tmp_path, monkeypatch):
    specs = tmp_path / "specs.json"
    specs.write_text("{}")
    path = write_config(tmp_path, '[run]\nruns_dir = "outputs"\nproblem_specs = "specs.json"\n')
    monkeypatch.chdir(ROOT)
    config = parse_config(["--config", str(path)])
    assert Path(config.runs_dir) == tmp_path / "outputs"
    assert Path(config.problem_specs) == specs
    override = parse_config(["--config", str(path), "--runs-dir", str(tmp_path / "override")])
    assert Path(override.runs_dir) == tmp_path / "override"


@pytest.mark.parametrize("text, message", [
    ('[gpu]\nprofile = "typo"', "Unknown GPU profile"),
    ('[gpu]\nhardware = "unknown"', "Unknown GPU"),
    ('[gpu]\ndevice = [0]', "Unknown \\[gpu\\]"),
    ('[gpu]\ndevices = [0, 0]', "gpu.devices"),
    ('[gpu]\ndevices = [-1]', "gpu.devices"),
    ('[gpu]\ndevices = [true]', "gpu.devices"),
    ('[gpu]\ndevices = ["0"]', "gpu.devices"),
    ('[gpu]\ndevices = []', "gpu.devices"),
    ('[gpu]\ndevices = "0,1"', "gpu.devices"),
    ('[gpu]\narchitecture = 8.6', "gpu.architecture"),
    ('[gpu]\narchitecture = "8.6;"', "Invalid GPU architecture"),
    ('[run]\ngeneration = "unknown"', "Invalid run.generation"),
    ('[run]\nparallel = "4"', "run.parallel must be int"),
    ('[run]\nparallel = true', "run.parallel must be int"),
    ('[run]\nresume = "false"', "run.resume must be bool"),
    ('[run]\nunknown = 1', "Unknown \\[run\\]"),
    ('[run]\nhardware = "H100"', "Put GPU settings in \\[gpu\\]"),
    ('[gup]\nprofile = "H100"', "Unknown config sections"),
    ('gpu = "H100"', "\\[gpu\\] must be a table"),
    ('run = "bad"', "\\[run\\] must be a table"),
    ('[gpu', "Cannot load config"),
])
def test_malformed_config_fails_early(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        parse_config(["--config", str(write_config(tmp_path, text))])


def test_missing_config_and_legacy_unknown_hardware_do_not_fall_back(tmp_path):
    with pytest.raises(ValueError, match="Cannot load config"):
        parse_config(["--config", str(tmp_path / "missing.toml")])
    with pytest.raises(ValueError, match="Unknown GPU"):
        parse_config(["--hardware", "unknown"])
    with pytest.raises(ValueError, match="Unknown GPU"):
        parse_config(["--gpu", "unknown"])
    config = parse_config([])
    assert not config.gpu_arch and not config.gpu and not config.hardware
    with pytest.raises(ValueError, match="Select a GPU"):
        config.require_gpu_target()


def test_cloud_target_must_be_explicit_for_gpu_without_cloud_preset():
    config = parse_config(["--gpu-profile", "RTX_A6000", "--evaluation", "modal"])
    with pytest.raises(ValueError, match="gpu.modal_type"):
        config.require_gpu_target()
    config = parse_config(["--gpu-profile", "H100", "--evaluation", "modal"])
    config.require_gpu_target()


def test_resolved_job_config_survives_missing_source_file(tmp_path):
    path = write_config(tmp_path, '''
[gpu]
hardware = "Custom_GPU"
architecture = "8.6"
devices = [0]
[run]
generation = "docker"
evaluation = "docker"
run_name = "parent"
''')
    config = parse_config(["--config", str(path), "--runs-dir", str(tmp_path)])
    directory = generation.prepare_job(config, 1, 0)
    path.unlink()
    settings = json.loads((directory / "job_config.json").read_text())
    child = RunConfig(**settings).validate()
    assert (child.hardware, child.gpu_arch, child.gpus) == ("Custom_GPU", "8.6", "0")
    assert child.evaluation == "local" and not child.gpu
    assert "--config" not in (directory / "verify.py").read_text()


def test_target_config_changes_invalidate_evaluation_and_generation(tmp_path):
    configs = [parse_config(["--gpu-profile", target, "--run-name", "same", "--runs-dir", str(tmp_path)])
               for target in ("RTX_A6000", "H100")]
    assert fingerprint(engine.task_payload(configs[0], 1, task="baseline")) != fingerprint(
        engine.task_payload(configs[1], 1, task="baseline"))
    assert generation.generation_signature(configs[0], 1, 0) != generation.generation_signature(configs[1], 1, 0)


def test_harbor_export_keeps_resolved_target_and_omits_empty_cloud_type(tmp_path):
    config = parse_config(["--gpu-profile", "RTX_A6000", "--run-name", "export", "--runs-dir", str(tmp_path)])
    task = harbor.write_task(tmp_path / "task", config, 90)
    parsed = tomllib.loads((task / "task.toml").read_text())
    assert "gpu_types" not in parsed["environment"]
    assert "gpu_types" not in parsed["verifier"]["environment"]
    manifest = json.loads((task / "tests/config.json").read_text())
    assert manifest["config"]["hardware"] == "RTX_A6000"
    assert manifest["config"]["gpu_arch"] == "Ampere"
    assert (task / "tests/repo/kb_workflows/hardware/gpu_profiles.toml").is_file()
    Task = pytest.importorskip("harbor.models.task.task").Task
    assert Task.is_valid_dir(task)
    assert Task(task).config.environment.gpus == 1


def test_missing_target_fails_before_any_run_or_cloud_work(tmp_path, capsys):
    assert cli.main(["--run-name", "missing", "--runs-dir", str(tmp_path)]) == 2
    assert "Select a GPU" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("example", ["a6000.toml", "h100-modal.toml", "a6000-one-shot.toml", "a6000-feedback.toml"])
def test_shipped_config_dry_runs_without_gpu_or_sdk_imports(tmp_path, example):
    script = '''
import sys
from kb_workflows.cli import main
code = main(sys.argv[1:])
assert not any(name in sys.modules for name in ("torch", "modal", "harbor", "minisweagent"))
raise SystemExit(code)
'''
    output = subprocess.run([sys.executable, "-c", script, "--config", str(ROOT / "configs" / example),
                             "--dry-run", "--runs-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    parsed = json.loads(output.stdout)
    assert parsed["config"]["gpu_arch"] and parsed["config"]["hardware"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("wrapper, example, expected", [
    ("batch_run.sh", "h100-modal.toml", "modal"),
    ("batch_run_modal.sh", "a6000.toml", "docker"),
])
def test_shell_wrappers_use_config_backend(wrapper, example, expected):
    output = subprocess.run(["bash", str(ROOT / wrapper), "--config", str(ROOT / "configs" / example), "--dry-run"],
                            env={**os.environ, "KB_PYTHON": sys.executable}, capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    assert json.loads(output.stdout)["config"]["evaluation"] == expected


def test_slurm_config_is_not_overridden_by_script_defaults(tmp_path):
    recorder = tmp_path / "record-python"
    recorder.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    recorder.chmod(0o755)
    keys = ["LEVEL", "START", "END", "RUN_NAME", "GPUS", "HARDWARE", "GPU_ARCH", "GPU_PROFILE", "GPU", "TIMEOUT",
            "ENABLE_NCU", "RESUME", "BASELINE", "AWS_PROFILE", "GENERATION", "EVALUATION"]
    environment = {k: v for k, v in os.environ.items() if k not in keys}
    environment.update(KB_CONFIG=str(ROOT / "configs/a6000.toml"), KB_REPO_DIR=str(ROOT),
                       KB_PYTHON=str(recorder), SLURM_JOB_ID="test", SLURM_GPUS_ON_NODE="2")
    output = subprocess.run(["bash", str(ROOT / "batch_run.sbatch")], env=environment, capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    assert json.loads(output.stdout) == ["scripts/run_batch.py", "--config", str(ROOT / "configs/a6000.toml")]
    environment.update(GPU_PROFILE="H100", GPUS="0,1", RESUME="true")
    output = subprocess.run(["bash", str(ROOT / "batch_run.sbatch")], env=environment, capture_output=True, text=True)
    args = json.loads(output.stdout)
    config = parse_config(args[1:])
    assert (config.hardware, config.gpu_arch, config.gpus, config.resume) == ("H100", "Hopper", "0,1", True)
    environment.pop("KB_CONFIG")
    output = subprocess.run(["bash", str(ROOT / "batch_run.sbatch")], env=environment, capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    config = parse_config(json.loads(output.stdout)[1:])
    assert (config.hardware, config.gpu_arch, config.gpus) == ("H100", "Hopper", "0,1")

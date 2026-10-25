"""Harbor task export and generation backend sharing the batch configuration."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from ..agents.generation import agent_environment, prompt_source
from ..benchmarks.reference_population import select_run_problems
from ..benchmarks.suites import selection_receipt
from ..common import ROOT, atomic_json, evaluator_digest, kernel_name, select_problems
from .transports import terminate

HARBOR_SOURCE = "b53b8134e1241686dca7759af188f987ecc48e8b"
CUDA_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"
BASE_IMAGE = f"""FROM {CUDA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.12 python3.12-venv python3-pip build-essential git curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
"""


def task_name(level, pid, sid):
    return f"l{level}-p{pid}-s{sid}"


def verification_timeout(config, pid):
    from ..evaluation.baseline_variants import variants_for

    count = len(variants_for(config, pid)) if config.baseline_suite else 1
    count += 2 + 2 * config.final_repeats
    if config.paired_timing:
        count += config.final_repeats or 1
    return count * config.timeout + 180


def _compose(device=None, *, restrictions=None):
    if device is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", device):
        raise ValueError("Harbor Docker device must be one numeric ID or GPU UUID")
    text = """services:
  main:
    build:
      context: .
    command: [sh, -c, "sleep infinity"]
"""
    if restrictions:
        text += f"""    network_mode: none
    read_only: true
    user: '65532:65532'
    cap_drop: [ALL]
    security_opt: [no-new-privileges:true]
    pids_limit: {restrictions['pids']}
    mem_limit: {restrictions['memory_mb']}m
    memswap_limit: {restrictions['memory_mb']}m
    cpus: {restrictions['cpus']}
    ipc: private
    tmpfs:
      - /tmp:rw,nosuid,nodev,size={restrictions['tmp_mb']}m,mode=1777
      - /app:rw,nosuid,nodev,size=64m,mode=1777
      - /tests/repo/cache:rw,nosuid,nodev,size={restrictions['tmp_mb']}m,uid=65532,gid=65532,mode=0700
      - /tests/repo/results:rw,nosuid,nodev,size=128m,uid=65532,gid=65532,mode=0700
"""
    if device is not None:
        text += f"""    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [{json.dumps(device)}]
              capabilities: [gpu]
"""
    return text


def write_task(directory, config, pid, sid=0, device=None):
    """Create a self-contained task; trusted evaluation code is verifier-only."""
    from ..benchmarks.reference_population import freeze_run

    population = freeze_run(config)
    if pid not in select_run_problems(config):
        raise ValueError("Harbor task is outside the declared reference population")
    if not config.hidden_tests:
        raise ValueError("Harbor tasks require hidden tests")
    reference = select_problems(config.level, problem_ids=[pid])[pid]
    hidden = ROOT / "hidden_tests" / f"level{config.level}" / f"{pid}_hidden.py"
    if not hidden.is_file():
        raise FileNotFoundError(f"Hidden tests missing: {hidden}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    environment, tests, solution = (directory / name for name in ("environment", "tests", "solution"))
    for path in (environment, tests, solution):
        path.mkdir()
    source = reference.read_text()
    if config.state_validation:
        from ..evaluation.state_policies import plan as state_plan

        state_plan(config.level, pid, source)
    if config.cache_validation:
        from ..evaluation.cache_recipes import plan as cache_plan

        cache_plan(config.level, pid, source)
    if config.level in {1, 2, 3}:
        from ..numerics.operator_input_policy import plan

        plan(config.level, pid, source, config.input_family_memory_mb)
    (environment / "model.py").write_text(prompt_source(source, config.level, pid))
    for target in (environment, tests):
        shutil.copy2(ROOT / "requirements.txt", target / "requirements.txt")
        shutil.copy2(ROOT / "LICENSE", target / "LICENSE")
        if (ROOT / "NOTICE.md").exists():
            shutil.copy2(ROOT / "NOTICE.md", target / "NOTICE.md")
    (environment / "Dockerfile").write_text(BASE_IMAGE + "COPY model.py LICENSE /app/\n")
    repo = tests / "repo"
    repo.mkdir()
    for name in ("src", "kb_workflows", "scripts", "KernelBench"):
        shutil.copytree(ROOT / name, repo / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "requirements.txt", repo / "requirements.txt")
    shutil.copy2(ROOT / "requirements-neuron-worker.txt", repo / "requirements-neuron-worker.txt")
    shutil.copy2(ROOT / "LICENSE", repo / "LICENSE")
    hidden_target = repo / "hidden_tests" / f"level{config.level}" / hidden.name
    hidden_target.parent.mkdir(parents=True)
    shutil.copy2(hidden, hidden_target)
    (tests / "Dockerfile").write_text(BASE_IMAGE + "COPY . /tests/\nENV PYTHONPATH=/tests/repo\n")
    restricted = config.isolation_mode == "restricted"
    from .isolation import allowed_hosts
    from .isolation import options as isolation_options

    if device is not None:
        (environment / "docker-compose.yaml").write_text(_compose(device))
    if device is not None or restricted:
        (tests / "docker-compose.yaml").write_text(_compose(device, restrictions=isolation_options(config) if restricted else None))
    child = replace(config, suite="production41" if config.level == 5 else "full", suite_role="scored", start=pid, end=pid, num_samples=1, run_name="harbor",
                    generation="local", evaluation="local", gpus="0", resume=False,
                    stage="evaluate", problem_specs="", hardware=config.hardware or config.gpu,
                    runs_dir="/logs/verifier/kernelbench-pro/runs", build_image=False,
                    profile="", bedrock=False, model="", small_model="", modal_secret="",
                    queue_url="http://127.0.0.1:8000", harbor_command="harbor")
    # The outer Harbor Docker sandbox owns these restrictions, rather than a nested Docker daemon.
    child.isolation_mode = "standard"
    child.reference_population = ""
    child.reference_population_scope = "run"
    # Reviewed models and calibration are verifier-only data, never agent inputs.
    if config.problem_specs:
        document = json.loads(Path(config.problem_specs).read_text())
        selected_spec = document.get(f"level{config.level}", {}).get(str(pid), {})
        atomic_json(tests / "problem_specs.json", {f"level{config.level}": {str(pid): selected_spec}})
        child.problem_specs = "/tests/problem_specs.json"
    if config.input_calibration:
        from ..numerics.input_calibration import subset

        atomic_json(tests / "input_calibration.json", subset(config.input_calibration, config.level, pid, config.precision, source, population=population))
        child.input_calibration = "/tests/input_calibration.json"
    if config.precision_calibration:
        from ..numerics.precision_calibration import subset_for_config

        atomic_json(tests / "precision_calibration.json", subset_for_config(config, pid, source))
        child.precision_calibration = "/tests/precision_calibration.json"
    if config.device_calibration:
        shutil.copy2(config.device_calibration, tests / "device_calibration.json")
        child.device_calibration = "/tests/device_calibration.json"
    if config.task_dimensions:
        shutil.copy2(config.task_dimensions, tests / "task_dimensions.json")
        child.task_dimensions = "/tests/task_dimensions.json"
    if config.determinism_policy_file:
        from ..evaluation.determinism import POLICY_PROTOCOL, read_policies

        policies = read_policies(config.determinism_policy_file)["tasks"]
        key = f"{config.level}/{pid}"
        atomic_json(tests / "determinism_policies.json", {"protocol": POLICY_PROTOCOL,
                    "tasks": {key: policies[key]} if key in policies else {}})
        child.determinism_policy_file = "/tests/determinism_policies.json"
    atomic_json(tests / "config.json", {"protocol": "kernelbench-pro-harbor-v1", "sample_id": sid,
                                       "evaluator_sha256": evaluator_digest(), "config": child.to_dict(),
                                       "restricted_policy": isolation_options(config) if restricted else None,
                                       "reward": config.harbor_reward})
    (tests / "test.sh").write_text("""#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
printf '0\\n' > /logs/verifier/reward.txt
rm -f /logs/verifier/reward.json
cd /tests/repo
exec python -m kb_workflows.runtime.harbor_verifier --manifest /tests/config.json \\
    --candidate /app/solution.py --logs /logs/verifier
""")
    (tests / "test.sh").chmod(0o755)
    (solution / "reference.py").write_text(source + "\nModelNew = Model\n")
    (solution / "solve.sh").write_text("#!/bin/bash\nset -euo pipefail\ncp /solution/reference.py /app/solution.py\n")
    (solution / "solve.sh").chmod(0o755)
    calibrated_instruction = ""
    if config.input_calibration:
        from ..numerics.input_calibration import record_for

        row = record_for(config.input_calibration, config.level, pid, config.precision, source)
        calibrated_instruction = (f"Nominal value distribution: {row['selected_recipe']}; frozen FP64 error policy: "
                                  f"{row['precision_contract']}. Shapes and discrete roles are unchanged. "
                                  "Both reference and candidate must qualify; hidden distributions remain separate.\n")
    if config.precision_calibration:
        from ..numerics.precision_calibration import for_config

        row = for_config(config, pid, source)
        calibrated_instruction += f"Effective device-fitted FP64 error policy: {row['policy']}; both reference and candidate must pass.\n"
    (directory / "instruction.md").write_text(f"""# KernelBench Pro: level {config.level}, problem {pid}

Implement ModelNew in /app/solution.py using custom GPU kernels.
Read /app/model.py for the reference interface and preserve its algorithm,
parameters, output structure, shape and dtype. The precision is {config.precision}.
Target GPU: {config.hardware or config.gpu}; architecture: {config.gpu_arch}.
{calibrated_instruction}Use a single self-contained Python file; inline CUDA and Triton are supported.
Do not modify model.py. Test your implementation with your own inputs.
The final verifier runs standard and multi-seed hidden correctness checks,
measures synchronized wall time and compares against PyTorch baselines.
It runs in a fresh environment; only /app/solution.py is submitted.
""")
    budget = verification_timeout(config, pid)
    resources = f"""build_timeout_sec = 1800.0
cpus = 4
memory_mb = {config.harbor_memory_mb}
storage_mb = 40960
gpus = 1
"""
    if config.gpu:
        resources += f"gpu_types = [{json.dumps(config.gpu)}]\n"
    (directory / "task.toml").write_text(f"""schema_version = "1.4"
artifacts = ["/app/solution.py"]

[task]
name = "kernelbench-pro/{task_name(config.level, pid, sid)}"
version = "1.0.0"
description = "GPU kernel generation with KernelBench Pro verification"

[metadata]
category = "gpu"
tags = ["cuda", "kernelbench-pro", "{config.precision}"]

[agent]
timeout_sec = {float(config.agent_timeout)}
{('network_mode = "allowlist"' + chr(10) + 'allowed_hosts = ' + json.dumps(allowed_hosts(config.harbor_agent_hosts))) if config.harbor_agent_hosts else ('network_mode = "no-network"' if restricted else '')}

[environment]
{resources}
[verifier]
timeout_sec = {float(budget)}
environment_mode = "separate"
network_mode = "no-network"

[verifier.environment]
{resources.replace('cpus = 4', 'cpus = ' + str(config.isolation_cpus)).replace('memory_mb = ' + str(config.harbor_memory_mb), 'memory_mb = ' + str(config.isolation_memory_mb)) if restricted else resources}""")
    return directory


def export_tasks(config):
    output = config.run_dir / "harbor_tasks"
    if output.exists():
        raise FileExistsError(f"Harbor export already exists: {output}; choose a new run name")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".harbor-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "tasks"
        staged.mkdir()
        tasks = []
        for pid in select_run_problems(config):
            for sid in range(config.num_samples):
                name = task_name(config.level, pid, sid)
                write_task(staged / name, config, pid, sid)
                tasks.append(name)
        staged.rename(output)
    atomic_json(config.run_dir / "harbor_export.json", {"tasks": tasks, "suite": selection_receipt(config), "harbor_source": HARBOR_SOURCE,
                                                       "evaluator_sha256": evaluator_digest(), "path": str(output)})
    return output


def run_harbor_agent(directory, config, pid, device=None, sid=0):
    """One Harbor attempt maps to one native (problem, sample) generation slot."""
    directory = Path(directory)
    from .harbor_budget import harbor_budget
    job_name = "trial-" + uuid.uuid4().hex[:12]
    task = write_task(directory / job_name, config, pid, sid=sid,
                      device=device if config.harbor_environment == "docker" else None)
    jobs = directory / "harbor_jobs"
    command = [config.harbor_command, "run", "-p", str(task), "-a", config.harbor_agent,
               "-e", config.harbor_environment, "--jobs-dir", str(jobs), "--job-name", job_name,
               "--n-attempts", "1", "--n-concurrent", "1"]
    if config.model:
        command += ["-m", config.model]
    with (directory / "harbor.log").open("w") as log:
        process = subprocess.Popen(command, env=runtime_environment(config),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=config.agent_timeout + 4800 + verification_timeout(config, pid))
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Harbor job timed out", "harbor_job": str(jobs / job_name),
                    "budget": harbor_budget(jobs / job_name)}
        finally:
            terminate(process)
    candidates = list((jobs / job_name).glob("*/artifacts/app/solution.py"))
    budget = harbor_budget(jobs / job_name)
    if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_file() or not candidates[0].stat().st_size:
        return {"status": "failed", "exit_code": code, "error": "Missing or ambiguous Harbor submission", "budget": budget}
    output = directory / "runs" / "agent" / kernel_name(config.level, pid)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(candidates[0].read_bytes())
    return {"status": "success" if code == 0 else "failed", "exit_code": code,
            "harbor_job": str(jobs / job_name), "budget": budget}


def runtime_environment(config):
    """Resolve host profiles to runtime credentials; never put them in task files."""
    environment = {**os.environ, **agent_environment(config)}
    if config.profile:
        import boto3

        session = boto3.Session(profile_name=config.profile)
        credentials = session.get_credentials()
        if credentials is None:
            raise ValueError("AWS profile did not provide credentials for Harbor")
        frozen = credentials.get_frozen_credentials()
        environment.update(AWS_ACCESS_KEY_ID=frozen.access_key, AWS_SECRET_ACCESS_KEY=frozen.secret_key)
        environment.pop("AWS_SESSION_TOKEN", None)
        if frozen.token:
            environment["AWS_SESSION_TOKEN"] = frozen.token
        if session.region_name:
            environment.update(AWS_REGION=session.region_name, AWS_DEFAULT_REGION=session.region_name)
        environment.pop("AWS_PROFILE", None)
    return environment

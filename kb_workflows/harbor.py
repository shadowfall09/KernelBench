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

from .common import ROOT, atomic_json, evaluator_digest, kernel_name, select_problems
from .generation import agent_environment, prompt_source
from .suites import selection_receipt
from .transports import terminate

HARBOR_SOURCE = "6c9d3d6609854375005a7b5c8f1906371d64c6ff"
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


def _compose(device):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", device):
        raise ValueError("Harbor Docker device must be one numeric ID or GPU UUID")
    return f"""services:
  main:
    build:
      context: .
    command: [sh, -c, "sleep infinity"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [{json.dumps(device)}]
              capabilities: [gpu]
"""


def write_task(directory, config, pid, sid=0, device=None):
    """Create a self-contained task; trusted evaluation code is verifier-only."""
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
    shutil.copy2(ROOT / "LICENSE", repo / "LICENSE")
    hidden_target = repo / "hidden_tests" / f"level{config.level}" / hidden.name
    hidden_target.parent.mkdir(parents=True)
    shutil.copy2(hidden, hidden_target)
    (tests / "Dockerfile").write_text(BASE_IMAGE + "COPY . /tests/\nENV PYTHONPATH=/tests/repo\n")
    if device is not None:
        for target in (environment, tests):
            (target / "docker-compose.yaml").write_text(_compose(device))
    child = replace(config, suite="full", suite_role="scored", start=pid, end=pid, num_samples=1, run_name="harbor",
                    generation="local", evaluation="local", gpus="0", resume=False,
                    stage="evaluate", problem_specs="", hardware=config.hardware or config.gpu,
                    runs_dir="/logs/verifier/kernelbench-pro/runs", build_image=False,
                    profile="", bedrock=False, model="", small_model="", modal_secret="",
                    queue_url="http://127.0.0.1:8000", harbor_command="harbor")
    atomic_json(tests / "config.json", {"protocol": "kernelbench-pro-harbor-v1", "sample_id": sid,
                                       "evaluator_sha256": evaluator_digest(), "config": child.to_dict(),
                                       "reward": config.harbor_reward})
    (tests / "test.sh").write_text("""#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
printf '0\\n' > /logs/verifier/reward.txt
rm -f /logs/verifier/reward.json
cd /tests/repo
exec python -m kb_workflows.harbor_verifier --manifest /tests/config.json \\
    --candidate /app/solution.py --logs /logs/verifier
""")
    (tests / "test.sh").chmod(0o755)
    (solution / "reference.py").write_text(source + "\nModelNew = Model\n")
    (solution / "solve.sh").write_text("#!/bin/bash\nset -euo pipefail\ncp /solution/reference.py /app/solution.py\n")
    (solution / "solve.sh").chmod(0o755)
    (directory / "instruction.md").write_text(f"""# KernelBench Pro: level {config.level}, problem {pid}

Implement ModelNew in /app/solution.py using custom GPU kernels.
Read /app/model.py for the reference interface and preserve its algorithm,
parameters, output structure, shape and dtype. The precision is {config.precision}.
Target GPU: {config.hardware or config.gpu}; architecture: {config.gpu_arch}.
Use a single self-contained Python file; inline CUDA and Triton are supported.
Do not modify model.py. Test your implementation with your own inputs.
The final verifier runs standard and multi-seed hidden correctness checks,
measures synchronized wall time and compares against PyTorch baselines.
It runs in a fresh environment; only /app/solution.py is submitted.
""")
    budget = (6 if config.baseline_suite else 3) * config.timeout + 180
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

[environment]
{resources}
[verifier]
timeout_sec = {float(budget)}
environment_mode = "separate"

[verifier.environment]
{resources}""")
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
        for pid in select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role):
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
            code = process.wait(timeout=config.agent_timeout + 4800 + 6 * config.timeout + 180)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Harbor job timed out", "harbor_job": str(jobs / job_name)}
        finally:
            terminate(process)
    candidates = list((jobs / job_name).glob("*/artifacts/app/solution.py"))
    if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_file() or not candidates[0].stat().st_size:
        return {"status": "failed", "exit_code": code, "error": "Missing or ambiguous Harbor submission"}
    output = directory / "runs" / "agent" / kernel_name(config.level, pid)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(candidates[0].read_bytes())
    return {"status": "success" if code == 0 else "failed", "exit_code": code,
            "harbor_job": str(jobs / job_name)}


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

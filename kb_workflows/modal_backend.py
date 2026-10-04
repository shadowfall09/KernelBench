"""Modal SDK is imported only when a cloud backend is selected."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from contextlib import contextmanager

from .common import ROOT, atomic_json, kernel_name

_SESSION = None


def _remote_evaluate(payload):
    from kb_workflows.transports import local_evaluate

    return local_evaluate(payload)


def make_image(agent=False):
    import modal

    image = (
        modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.10")
        .apt_install("build-essential", "git", "curl", "jq")
        .pip_install_from_requirements(str(ROOT / "requirements.txt"))
        .env({"PYTHONPATH": "/app/KernelBench", "PYTHONUNBUFFERED": "1"})
    )
    if agent:
        image = image.run_commands("curl -fsSL https://claude.ai/install.sh | bash").env(
            {"PATH": "/root/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"}
        )
    for name in ("src", "KernelBench", "hidden_tests", "scripts", "kb_workflows"):
        image = image.add_local_dir(ROOT / name, f"/app/KernelBench/{name}", ignore=["**/__pycache__/**"])
    image = image.add_local_file(ROOT / "requirements.txt", "/app/KernelBench/requirements.txt")
    return image


@contextmanager
def modal_session(config):
    global _SESSION
    import modal

    app = modal.App("kernelbench-verified")
    evaluate = app.function(image=make_image(), gpu=config.gpu, timeout=config.timeout + 60, cpu=4.0, memory=32768)(
        _remote_evaluate
    )
    agent_image = make_image(agent=True) if config.generation == "modal" else None
    with modal.enable_output(), app.run():
        _SESSION = (app, evaluate, agent_image)
        try:
            yield
        finally:
            _SESSION = None


def modal_evaluate(payload, config):
    if _SESSION is None:
        raise RuntimeError("Modal evaluation must run within modal_session")
    return _SESSION[1].remote(payload)


def sandbox_secrets(config):
    import modal

    if config.modal_secret:
        return [modal.Secret.from_name(config.modal_secret)]
    names = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    }
    values = {key: value for key, value in os.environ.items() if key in names}
    if config.profile:
        # Resolve SSO/profile credentials at runtime; never bake ~/.aws into an image.
        result = subprocess.run(
            ["aws", "configure", "export-credentials", "--profile", config.profile, "--format", "process"],
            capture_output=True,
            text=True,
            check=True,
        )
        credentials = json.loads(result.stdout)
        for source, target in (
            ("AccessKeyId", "AWS_ACCESS_KEY_ID"),
            ("SecretAccessKey", "AWS_SECRET_ACCESS_KEY"),
            ("SessionToken", "AWS_SESSION_TOKEN"),
        ):
            if credentials.get(source):
                values[target] = credentials[source]
    return [modal.Secret.from_dict(values)] if values else []


def run_modal_agent(directory, config, pid):
    import modal

    from .generation import agent_environment

    if _SESSION is None:
        raise RuntimeError("Modal generation must run within modal_session")
    job = json.loads((directory / "job_config.json").read_text())
    job["profile"] = ""  # Runtime credentials/secrets do not need an AWS profile file.
    environment = agent_environment(config)
    environment.pop("AWS_PROFILE", None)
    sandbox = modal.Sandbox.create(
        app=_SESSION[0],
        image=_SESSION[2],
        gpu=config.gpu,
        env=environment,
        secrets=sandbox_secrets(config),
        timeout=config.agent_timeout + 120,
        workdir="/app/KernelBench",
        cpu=4.0,
        memory=32768,
    )
    result = {"status": "failed"}
    try:
        mkdir = sandbox.exec("mkdir", "-p", "/work/runs/agent")
        mkdir.wait()
        for name, content in (
            ("job_config.json", json.dumps(job)),
            ("model.py", (directory / "model.py").read_text()),
            ("verify.py", (directory / "verify.py").read_text()),
        ):
            with sandbox.open(f"/work/{name}", "w") as stream:
                stream.write(content)
        command = [
            "python",
            "/app/KernelBench/scripts/run_agent.py",
            "--config",
            "/work/job_config.json",
            "--directory",
            "/work",
        ]
        process = sandbox.exec(
            "bash", "-c", shlex.join(command) + " > /work/sandbox.log 2>&1", timeout=config.agent_timeout + 60
        )
        process.wait()
    finally:
        # Download before termination; partial kernels/logs remain useful after agent failure.
        for name in ("agent_result.json", "agent.log", "sandbox.log", f"runs/agent/{kernel_name(config.level, pid)}"):
            try:
                with sandbox.open(f"/work/{name}", "r") as stream:
                    content = stream.read()
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                if name == "agent_result.json":
                    result = json.loads(content)
            except (FileNotFoundError, modal.exception.Error):
                continue
        sandbox.terminate()
    atomic_json(directory / "agent_result.json", result)
    return result

"""Modal SDK is imported only when a cloud backend is selected."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import subprocess
import time
from contextlib import contextmanager, nullcontext

from ..agents.resource_budgets import CLEANUP_SECONDS, record_device_lease, validate_device_lease
from ..common import ROOT, atomic_json, failure, fingerprint, kernel_name

_SESSION = None
LEASE_PROTOCOL = "kernelbench_pro_modal_lease_v1"


def _remote_evaluate(payload, leases=None):
    from .transports import local_evaluate

    lease = validate_device_lease(payload)
    if "worker_python" in payload or payload["eval_config"].get("device_backend", "cuda") != "cuda":
        raise ValueError("Modal verification requires CUDA and the server-owned interpreter")
    signature = fingerprint(payload)

    @record_device_lease
    def execute(payload):
        if lease:
            if leases is None:
                raise ValueError("Modal device leases require a shared admission store")
            if leases.put(lease["nonce"], signature, skip_if_exists=True) is not True:
                return failure("Modal device lease was already admitted; execution will not be replayed",
                               resource_usage_unknown=True)
        return local_evaluate(payload)

    result = execute(payload)
    if lease is None:
        return result
    return {"protocol": LEASE_PROTOCOL, "request_sha256": signature, "nonce": lease["nonce"],
            "device_backend": "cuda", "result": result}


def make_image(agent=False):
    import modal

    image = (
        modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
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
    image = image.add_local_file(ROOT / "requirements-neuron-worker.txt", "/app/KernelBench/requirements-neuron-worker.txt")
    return image


@contextmanager
def modal_session(config):
    global _SESSION
    import modal

    app = modal.App("kernelbench-pro")
    evaluate = app.function(image=make_image(), gpu=config.gpu, timeout=config.timeout + 60, cpu=4.0, memory=32768, retries=0)(
        _remote_evaluate
    )
    agent_image = make_image(agent=True) if config.generation == "modal" else None
    lease_context = modal.Dict.ephemeral() if config.agent_device_seconds else nullcontext(None)
    with modal.enable_output(), lease_context as leases, app.run():
        previous, _SESSION = _SESSION, (app, evaluate, agent_image, leases)
        try:
            yield
        finally:
            _SESSION = previous


def modal_evaluate(payload, config):
    if _SESSION is None:
        raise RuntimeError("Modal evaluation must run within modal_session")
    lease = validate_device_lease(payload)
    if lease is None:
        return _SESSION[1].remote(payload)
    # Queue/cold-start waiting is bounded separately from the remote worker's
    # monotonic receipt. A lost submission may still have admitted work: never
    # resubmit or refund unknown usage, even if cancellation was acknowledged.
    signature = fingerprint(payload)
    session = _SESSION
    deadline = time.monotonic() + min(config.queue_wait_timeout, lease["execution_timeout_seconds"])

    async def invoke():
        call = None
        try:
            function = session[1].with_options(timeout=max(1, math.ceil(lease["reserved_seconds"])), retries=0)
            call = await asyncio.wait_for(function.spawn.aio(payload, session[3]), timeout=max(0, deadline - time.monotonic()))
            remaining = max(0, deadline - time.monotonic())
            response = await asyncio.wait_for(call.get.aio(timeout=remaining), timeout=remaining)
            if time.monotonic() > deadline:
                raise TimeoutError("Modal development wait deadline expired")
            if (not isinstance(response, dict) or response.get("protocol") != LEASE_PROTOCOL
                    or response.get("request_sha256") != signature or response.get("nonce") != lease["nonce"]
                    or response.get("device_backend") != "cuda" or not isinstance(response.get("result"), dict)):
                raise ValueError("Modal response does not bind the requested task and device lease")
            return response["result"]
        except Exception as exc:
            result = failure(str(exc), error_name=type(exc).__name__, resource_usage_unknown=True)
            if call is not None:
                try:
                    await asyncio.wait_for(call.cancel.aio(terminate_containers=True), timeout=CLEANUP_SECONDS)
                    result["metadata"]["modal_cancel_requested"] = True
                except Exception as cancel_error:
                    result["metadata"]["modal_cancel_error"] = type(cancel_error).__name__
            return result

    return asyncio.run(invoke())


def sandbox_secrets(config):
    import modal

    if config.modal_secret:
        return [modal.Secret.from_name(config.modal_secret)]
    names = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "GEMINI_API_KEY",
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

    from ..agents.generation import agent_environment

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
        for name in ("agent_result.json", "agent.log", "sandbox.log", "trajectory.json", "development_history.json",
                     "best_candidate.json", f"runs/agent/{kernel_name(config.level, pid)}"):
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

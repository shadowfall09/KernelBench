"""Modal SDK is imported only when a cloud backend is selected."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath

from ..agents.resource_budgets import CLEANUP_SECONDS, record_device_lease, validate_device_lease
from ..common import ROOT, atomic_json, evaluator_digest, failure, fingerprint, kernel_name, read_json

_SESSION = None
LEASE_PROTOCOL = "kernelbench_pro_modal_lease_v1"
GENERATION_PROTOCOL = "kernelbench_pro_modal_generation_v1"
AGENT_INPUTS = ("model.py", "verify.py", "input_calibration.json", "precision_calibration.json", "determinism_policy.json")
AGENT_ARTIFACTS = ("agent.log", "trajectory.json", "development_history.json", "best_candidate.json", "claude.stderr.log")


def generation_binding(job, inputs, nonce):
    """Bind the actual sandbox job and uploaded text, without loading Modal."""
    if (not isinstance(nonce, str) or len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce)
            or job.get("generation") != "modal" or job.get("evaluation") != "local" or job.get("gpus") != "0"
            or job.get("device_backend", "cuda") != "cuda" or job.get("profile") != ""
            or not {"model.py", "verify.py"} <= set(inputs) <= set(AGENT_INPUTS)
            or any(not isinstance(value, str) for value in inputs.values())):
        raise ValueError("Invalid native Modal generation admission")
    return {"protocol": GENERATION_PROTOCOL, "nonce": nonce, "input_files": sorted(inputs),
            "evaluator_sha256": evaluator_digest(), "request_sha256": fingerprint({"job": job, "inputs": inputs})}


def generation_result_sha(result):
    # Parent bookkeeping is attached after the remote controller has completed.
    ignored = {"modal_generation_receipt", "modal_cleanup", "elapsed", "signature", "kernel_sha256"}
    return fingerprint({key: value for key, value in result.items() if key not in ignored})


def admitted_generation(directory, job, admission, invoke):
    """One controller per sandbox, even if the exec RPC is replayed."""
    directory = Path(directory)
    inputs = {name: (directory / name).read_text() for name in AGENT_INPUTS if (directory / name).is_file()}
    binding = generation_binding(job, inputs, admission.get("nonce"))
    if binding != admission:
        raise ValueError("Modal generation inputs, configuration or evaluator changed")
    with (directory / "generation_admitted.json").open("x") as marker:
        json.dump(binding, marker)
    result = invoke()
    limit = job["isolation_report_mb"] * 1024**2
    output = f"runs/agent/{kernel_name(job['level'], job['start'])}"
    paths = [directory / name for name in (*AGENT_ARTIFACTS, output)]
    for name in ("rounds", "model_calls"):
        root = directory / name
        if root.is_symlink():
            raise ValueError("Modal generation artifacts cannot use symlink directories")
        for path in root.rglob("*"):
            if len(paths) >= 10_000:
                raise ValueError("Too many Modal generation artifacts")
            paths.append(path)
    artifacts, total = {}, 0
    for path in paths:
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(directory)):
            raise ValueError("Modal generation artifacts cannot use symlinks")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > limit - total:
            raise ValueError("Modal generation artifacts exceed the report byte limit")
        with path.open("rb") as stream:
            data = stream.read(size + 1)
        if len(data) != size:
            raise ValueError("Modal generation artifact changed during collection")
        total += size
        artifacts[path.relative_to(directory).as_posix()] = {"bytes": size, "sha256": hashlib.sha256(data).hexdigest()}
    receipt = {"binding": binding, "result_sha256": generation_result_sha(result), "artifacts": artifacts,
               "physical_preemption_certified": False}
    receipt["receipt_sha256"] = fingerprint(receipt)
    result["modal_generation_receipt"] = receipt
    return result


def generation_receipt(result, binding, job):
    receipt = result.get("modal_generation_receipt", {})
    if (not isinstance(receipt, dict) or receipt.get("binding") != binding
            or receipt.get("receipt_sha256") != fingerprint({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or receipt.get("result_sha256") != generation_result_sha(result)
            or receipt.get("physical_preemption_certified") is not False):
        raise ValueError("Modal generation result does not bind the admitted controller")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) > 10_000:
        raise ValueError("Invalid Modal generation artifact inventory")
    total = 0
    output = f"runs/agent/{kernel_name(job['level'], job['start'])}"
    allowed = {*AGENT_ARTIFACTS, output}
    for name, row in artifacts.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or not path.parts or name not in allowed and not (len(path.parts) > 1 and path.parts[0] in {"rounds", "model_calls"})
                or not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] < 0
                or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in row["sha256"])):
            raise ValueError("Invalid Modal generation artifact identity")
        total += row["bytes"]
    if total > job["isolation_report_mb"] * 1024**2:
        raise ValueError("Modal generation artifacts exceed the report byte limit")
    return artifacts


def validate_generation(result, directory):
    """Recheck the transferred controller/ledger artifacts during scoring."""
    directory = Path(directory)
    job = read_json(directory / "job_config.json")
    job["profile"] = ""
    inputs = {name: (directory / name).read_text() for name in AGENT_INPUTS if (directory / name).is_file()}
    binding = generation_binding(job, inputs, result["modal_generation_receipt"]["binding"]["nonce"])
    artifacts = generation_receipt(result, binding, job)
    cleanup = result.get("modal_cleanup", {})
    output = f"runs/agent/{kernel_name(job['level'], job['start'])}"
    if (cleanup.get("binding") != binding or cleanup.get("termination_wait_completed") is not True
            or type(cleanup.get("returncode")) is not int
            or result.get("status") == "success" and (output not in artifacts or artifacts[output]["bytes"] == 0)):
        raise ValueError("Modal generation lacks completed termination or selected source")
    for name, expected in artifacts.items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Modal generation artifact is outside its job")
        with path.open("rb") as stream:
            data = stream.read(expected["bytes"] + 1)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("Modal generation artifact is missing or changed")
    return binding


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
    lease_context = modal.Dict.ephemeral() if (config.agent_device_seconds or config.agent_gpu_busy_seconds) and config.generation != "modal" else nullcontext(None)
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
    job = read_json(directory / "job_config.json")
    job["profile"] = ""  # Credentials come from runtime secrets, not a host profile file.
    inputs = {name: (directory / name).read_text() for name in AGENT_INPUTS if (directory / name).is_file()}
    binding = generation_binding(job, inputs, uuid.uuid4().hex)
    environment = agent_environment(config)
    environment.pop("AWS_PROFILE", None)
    session = _SESSION
    limit = config.isolation_report_mb * 1024**2
    # This bounds cloud orchestration, not worker leases or GPU allocation billing.
    window = config.agent_timeout + 120
    deadline = time.monotonic() + window

    async def execute():
        sandbox, result = None, {"status": "error", "budget": None, "resource_receipt_incomplete": True,
                                 "modal_cleanup": {"binding": binding, "termination_wait_completed": False}}
        complete = False
        output = f"runs/agent/{kernel_name(config.level, pid)}"

        async def bounded(method, *args, **kwargs):
            return await asyncio.wait_for(method.aio(*args, **kwargs), timeout=max(0, deadline - time.monotonic()))

        async def read(name, maximum):
            info = await bounded(sandbox.filesystem.stat, "/work/" + name)
            if type(info.size) is not int or not 0 <= info.size <= maximum:
                raise ValueError("Modal generation artifact exceeds its byte limit")
            data = await bounded(sandbox.filesystem.read_bytes, "/work/" + name)
            if not isinstance(data, bytes) or len(data) != info.size:
                raise ValueError("Modal generation artifact changed during transfer")
            return data

        def save(name, data):
            target = directory / name
            if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Modal generation artifact is outside its local job")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        try:
            sandbox = await bounded(modal.Sandbox.create, app=session[0], image=session[2], gpu=config.gpu,
                                    env=environment, secrets=sandbox_secrets(config), timeout=window,
                                    workdir="/app/KernelBench", cpu=4.0, memory=32768)
            await bounded(sandbox.filesystem.make_directory, "/work/runs/agent")
            for name, content in {**inputs, "job_config.json": json.dumps(job),
                                  "generation_admission.json": json.dumps(binding)}.items():
                await bounded(sandbox.filesystem.write_text, content, "/work/" + name)
            command = ["python", "/app/KernelBench/scripts/run_agent.py", "--config", "/work/job_config.json",
                       "--directory", "/work", "--admission", "/work/generation_admission.json"]
            process = await bounded(sandbox.exec, "bash", "-c", shlex.join(command) + " > /work/sandbox.log 2>&1",
                                    timeout=max(1, min(config.agent_timeout + 60, math.ceil(deadline - time.monotonic()))))
            await bounded(process.wait)
            remote = json.loads(await read("agent_result.json", limit))
            artifacts = generation_receipt(remote, binding, job)
            if remote.get("status") == "success" and (output not in artifacts or artifacts[output]["bytes"] == 0):
                raise ValueError("Modal generation success lacks its selected source")
            for name in sorted(artifacts, key=lambda name: (name != output, name)):
                expected = artifacts[name]
                data = await read(name, expected["bytes"])
                if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    raise ValueError("Modal generation artifact bytes differ from the controller receipt")
                save(name, data)
            if time.monotonic() > deadline:
                raise TimeoutError("Modal generation orchestration deadline expired")
            result = remote
            complete = True
        except Exception as exc:
            result.update(status="error", budget=None, resource_receipt_incomplete=True,
                          error=f"{type(exc).__name__}: {exc}"[:1000])
        finally:
            if sandbox is not None:
                if not complete:
                    # Retain useful but unqualified partial work only while the
                    # original deadline still allows it. Never reset the clock.
                    for name in (output, "best_candidate.json", "agent.log", "sandbox.log"):
                        try:
                            save(name, await read(name, limit))
                            result["partial_artifacts_unverified"] = True
                        except Exception:
                            pass
                try:
                    code = await asyncio.wait_for(sandbox.terminate.aio(wait=True), timeout=CLEANUP_SECONDS)
                    if type(code) is not int:
                        raise ValueError("Modal sandbox termination has no terminal return code")
                    result["modal_cleanup"] = {"binding": binding, "termination_wait_completed": True, "returncode": code}
                except Exception as exc:
                    result.update(status="error", budget=None, resource_receipt_incomplete=True,
                                  error=f"Modal sandbox termination unconfirmed: {type(exc).__name__}"[:1000])
                    result["modal_cleanup"] = {"binding": binding, "termination_wait_completed": False}
        atomic_json(directory / "agent_result.json", result)
        return result

    return asyncio.run(execute())

"""Parent-owned Docker lifecycle for restricted CUDA verification.

No container is created by import, preview, or the CPU regression suite.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from ..common import atomic_json, failure
from .isolation import create_command, inspect_issues, inspect_snapshot, policy, read_result, seal

MAX_CONTROL_OUTPUT = 16 * 1024**2


def invoke(command, *, timeout):
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        data, _ = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.communicate()
        raise TimeoutError("Docker control request timed out") from None
    if len(data) > MAX_CONTROL_OUTPUT:
        raise ValueError("Docker control output exceeded the retained bound")
    return {"exit_code": child.returncode, "output": data.decode(errors="replace")}


def evaluate(payload, config, device, log_path, *, run=None, clock=time.monotonic):
    injected, run = run is not None, run or invoke
    spec = policy(payload["eval_config"]["isolation"])
    if payload["eval_config"].get("device_backend", "cuda") != "cuda":
        raise ValueError("Restricted Docker worker currently requires CUDA")
    started = clock()
    deadline = started + payload["eval_config"]["timeout"]
    name = "kb-restricted-" + uuid.uuid4().hex
    created, cleaned, timed_out, raw, error = False, False, False, None, None
    observations, image, command, exit_code = {}, None, None, None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def control(args, *, cleanup=False, maximum=10):
        remaining = deadline - clock() - (0 if cleanup else 5)
        if remaining <= 0:
            raise TimeoutError("Restricted verifier exhausted its request budget")
        response = run(["docker", *args], timeout=min(maximum, remaining))
        if response.get("exit_code") != 0:
            raise RuntimeError("Docker control command did not complete: " + args[0])
        return response["output"]

    with tempfile.TemporaryDirectory(prefix="kb-restricted-") as temporary:
        directory = Path(temporary)
        request, output = directory / "request", directory / "output"
        request.mkdir(mode=0o755)
        output.mkdir(mode=0o777)
        output.chmod(0o777)
        atomic_json(request / "request.json", payload)
        (request / "request.json").chmod(0o444)
        request.chmod(0o555)
        try:
            image = control(["image", "inspect", config.docker_image, "--format", "{{.Id}}"], maximum=5).strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
                raise ValueError("Verifier image is not locally available by immutable ID")
            command = create_command(spec, image=image, name=name, device=device, request_dir=request, output_dir=output)
            identifier = control(command[1:]).strip()
            if not re.fullmatch(r"[0-9a-f]{64}", identifier):
                raise ValueError("Docker did not return a container identity")
            created = True
            document = json.loads(control(["inspect", identifier], maximum=5))
            if not isinstance(document, list) or len(document) != 1:
                raise ValueError("Missing created-container inspection")
            observations["before"] = inspect_snapshot(document[0])
            if observations["before"]["id"] != identifier or inspect_issues(observations["before"], spec,
                    image=image, device=device, request_dir=request, output_dir=output, command=command):
                raise ValueError("Created Docker container does not enforce the requested policy")
            control(["start", identifier])
            value = control(["wait", identifier], maximum=max(0, deadline - clock() - 5)).strip()
            if not re.fullmatch(r"[0-9]+", value):
                raise ValueError("Docker wait did not return a process exit status")
            exit_code = int(value)
            document = json.loads(control(["inspect", identifier], maximum=5))
            observations["after"] = inspect_snapshot(document[0])
            if log_path is not None:
                text = control(["logs", "--tail", "100", identifier], maximum=2)
                Path(log_path).write_text(text[-1024**2:])
            state = observations["after"]["state"]
            if exit_code != 0 or state.get("Running") is not False or state.get("ExitCode") != 0 or state.get("OOMKilled"):
                raise ValueError("Restricted verifier exited, timed out or was killed without a successful result")
            raw = read_result(output / "result.json", spec["report_mb"])
            if not isinstance(raw, dict):
                raise ValueError("Restricted verifier returned a non-object")
        except TimeoutError as exc:
            timed_out, error = True, str(exc)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if not created and command is not None:
                # create may have succeeded while its client timed out. Reconcile only our nonce-labelled container.
                try:
                    reply = run(["docker", "inspect", name], timeout=max(0.01, min(1, deadline - clock())))
                    rows = json.loads(reply["output"]) if reply.get("exit_code") == 0 else []
                    if rows and (rows[0].get("Config", {}).get("Labels") or {}).get("kernelbench-pro.owner") == name:
                        identifier = rows[0]["Id"]
                        created = bool(re.fullmatch(r"[0-9a-f]{64}", identifier))
                except (ValueError, KeyError, TypeError, OSError, RuntimeError, TimeoutError):
                    pass
            if created:
                try:
                    control(["rm", "--force", "--volumes", identifier], cleanup=True, maximum=3)
                    # A successful rm is independently checked rather than inferred from a CLI exit code.
                    reply = run(["docker", "container", "ls", "--all", "--quiet", "--filter", f"id={identifier}"],
                                timeout=max(0.01, min(2, deadline - clock())))
                    cleaned = reply.get("exit_code") == 0 and reply.get("output", "").strip() == ""
                except (OSError, RuntimeError, ValueError, TimeoutError):
                    cleaned = False
            else:
                cleaned = True
        document = {"policy": spec, "image": image, "device": str(device), "request_dir": str(request), "output_dir": str(output),
            "command": command, **observations, "cleanup_confirmed": cleaned, "timed_out": timed_out, "exit_code": exit_code,
            "host_namespace_inodes": {key: Path(f"/proc/self/ns/{key}").stat().st_ino for key in ("pid", "net", "mnt", "ipc")},
            "simulated_execution": injected, "request_wall_seconds": clock() - started,
            "scope": "restricted_Docker_policy_and_worker_observations; not_host_or_GPU_exclusivity_or_escape_certification"}
        if error or not cleaned or raw is None:
            return failure(error or "Restricted container cleanup/result is unverified", execution_envelope=document)
        result = seal(raw, document, payload)
        from .isolation import validate

        try:
            validate(result, payload, allow_simulated=injected)
        except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
            return failure(f"Restricted verifier qualification failed: {exc}", execution_envelope=document)
        return result

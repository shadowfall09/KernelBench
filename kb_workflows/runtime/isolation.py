"""Restricted verifier policies and source-bound observations, without Docker imports.

Policy observations are deliberately not an OS/GPU security certification.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import socket
import stat
from pathlib import Path

from ..common import evaluator_digest, fingerprint

PROTOCOL = "kernelbench_pro_execution_envelope_v1"
USER = "65532:65532"
ROOT = "/app/KernelBench"
DEFAULTS = {"mode": "standard", "cpus": 4, "memory_mb": 65536, "pids": 512, "tmp_mb": 16384, "report_mb": 64}
PARENT_FIELDS = ("workflow_signature", "evaluation_config", "device_lease_receipt", "execution_envelope")


def allowed_hosts(value):
    if not isinstance(value, str):
        raise ValueError("Harbor agent hosts require a comma-separated hostname/IP/CIDR list")
    if not value:
        return []
    hosts = []
    for host in value.split(","):
        host = host.strip().lower()
        try:
            normalized = str(ipaddress.ip_network(host, strict=True)) if "/" in host else str(ipaddress.ip_address(host))
        except ValueError:
            normalized = host
            domain = host.removeprefix("*.")
            if (len(domain) > 253 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+", domain)
                    or any(len(label) > 63 for label in domain.split("."))):
                raise ValueError("Agent network hosts must be explicit hostnames/IP/CIDRs, without schemes, paths or ports")
        if normalized in hosts:
            raise ValueError("Duplicate agent network host")
        hosts.append(normalized)
    return hosts


def options(config):
    return {name: getattr(config, "isolation_" + name) for name in DEFAULTS}


def policy(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS) or value.get("mode") not in {"standard", "restricted"}:
        raise ValueError("Invalid verifier isolation policy")
    if any(type(value[key]) is not int or value[key] < 1 for key in DEFAULTS if key != "mode"):
        raise ValueError("Verifier CPU/memory/PID/tmp/report limits require positive integers")
    if value["report_mb"] > 256 or value["pids"] > 65536:
        raise ValueError("Verifier report/PID limit exceeds supported bounds")
    return value


def enabled(payload):
    return payload.get("eval_config", {}).get("isolation", DEFAULTS).get("mode") == "restricted"


def normalized(result):
    value = copy.deepcopy(result)
    value.pop("sample_id", None)
    metadata = value.get("metadata", {})
    for key in PARENT_FIELDS:
        metadata.pop(key, None)
    return value


def create_command(spec, *, image, name, device, request_dir, output_dir):
    policy(spec)
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)", str(device)):
        raise ValueError("Restricted verification requires one device ID or GPU/MIG UUID")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image) or not re.fullmatch(r"kb-restricted-[0-9a-f]{32}", name):
        raise ValueError("Restricted worker image/name is not frozen")
    request_dir, output_dir = Path(request_dir).resolve(), Path(output_dir).resolve()
    if any("," in str(path) for path in (request_dir, output_dir)):
        raise ValueError("Docker bind paths containing commas are unsupported")
    limits = spec["memory_mb"] * 1024**2
    command = ["docker", "create", "--pull", "never", "--name", name, "--gpus", f"device={device}",
        "--label", "kernelbench-pro.owner=" + name,
        "--network", "none", "--read-only", "--user", USER, "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
        "--ipc", "private", "--cgroupns", "private", "--cpus", str(spec["cpus"]), "--memory", str(limits),
        "--memory-swap", str(limits), "--pids-limit", str(spec["pids"]), "--shm-size", "67108864",
        "--ulimit", "core=0:0", "--no-healthcheck", "--restart", "no",
        "--log-driver", "local", "--log-opt", "max-size=10m", "--log-opt", "max-file=1",
        "--mount", f"type=bind,src={request_dir},dst=/request,readonly",
        "--mount", f"type=bind,src={output_dir},dst=/output",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,size={spec['tmp_mb']}m,mode=1777",
        "--tmpfs", f"{ROOT}/cache:rw,nosuid,nodev,size={spec['tmp_mb']}m,uid=65532,gid=65532,mode=0700",
        "--workdir", ROOT, "--entrypoint", "/usr/bin/env", image]
    return [*command, "-i", "PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin", "HOME=/tmp/kb-home",
        "PYTHONPATH=" + ROOT, "PYTHONDONTWRITEBYTECODE=1", "XDG_CACHE_HOME=/tmp/cache", "CUDA_HOME=/usr/local/cuda",
        "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor", "TRITON_CACHE_DIR=/tmp/triton", "CUDA_CACHE_PATH=/tmp/cuda",
        "CUDA_VISIBLE_DEVICES=0", "/opt/venv/bin/python", "-B", "-s", "-m", "kb_workflows.runtime.isolated_worker",
        "/request/request.json", "/output/result.json"]


def inspect_snapshot(document):
    """Retain configuration relevant to enforcement, never image/host env secrets."""
    host, config = document.get("HostConfig", {}), document.get("Config", {})
    return {"id": document.get("Id"), "image": document.get("Image"),
        "host": {key: host.get(key) for key in ("NetworkMode", "ReadonlyRootfs", "Privileged", "CapAdd", "CapDrop", "SecurityOpt",
            "PidMode", "IpcMode", "UTSMode", "UsernsMode", "CgroupnsMode", "NanoCpus", "Memory", "MemorySwap", "PidsLimit",
            "DeviceRequests", "Devices", "Tmpfs", "PortBindings", "LogConfig", "RestartPolicy")},
        "config": {**{key: config.get(key) for key in ("User", "WorkingDir", "Entrypoint", "Cmd", "Healthcheck")},
                   "owner": (config.get("Labels") or {}).get("kernelbench-pro.owner")},
        "mounts": [{key: row.get(key) for key in ("Type", "Source", "Destination", "RW", "Propagation")}
                   for row in document.get("Mounts", [])],
        "state": {key: document.get("State", {}).get(key) for key in ("Running", "ExitCode", "OOMKilled", "Error")}}


def inspect_issues(document, spec, *, image, device, request_dir, output_dir, command):
    policy(spec)
    host, config = document["host"], document["config"]
    expected = {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False, "PidMode": "", "IpcMode": "private",
        "UTSMode": "", "CgroupnsMode": "private", "NanoCpus": spec["cpus"] * 10**9,
        "Memory": spec["memory_mb"] * 1024**2, "MemorySwap": spec["memory_mb"] * 1024**2, "PidsLimit": spec["pids"]}
    issues = [key for key, value in expected.items() if host.get(key) != value]
    if host.get("UsernsMode") not in ("", None) or host.get("CapAdd") or host.get("CapDrop") != ["ALL"]:
        issues.append("capabilities_or_user_namespace")
    if host.get("SecurityOpt") not in (["no-new-privileges=true"], ["no-new-privileges"]):
        issues.append("security_options")
    if document.get("image") != image or config.get("User") != USER or config.get("WorkingDir") != ROOT:
        issues.append("frozen_image_or_user")
    if config.get("owner") != command[command.index("--name") + 1]:
        issues.append("container_owner")
    if config.get("Entrypoint") != ["/usr/bin/env"] or config.get("Cmd") != command[command.index(image) + 1:]:
        issues.append("frozen_worker_command")
    if config.get("Healthcheck", {}).get("Test") != ["NONE"]:
        issues.append("image_healthcheck")
    requests = host.get("DeviceRequests") or []
    if (len(requests) != 1 or requests[0].get("DeviceIDs") != [str(device)] or requests[0].get("Count") not in (0, None)
            or requests[0].get("Driver") not in ("", "nvidia") or requests[0].get("Capabilities") != [["gpu"]]
            or host.get("Devices")):
        issues.append("device_allocation")
    expected_binds = {"/request": (str(Path(request_dir).resolve()), False), "/output": (str(Path(output_dir).resolve()), True)}
    binds = {row["Destination"]: (row["Source"], row["RW"]) for row in document["mounts"] if row["Type"] == "bind"}
    if binds != expected_binds or len([row for row in document["mounts"] if row["Type"] == "bind"]) != 2:
        issues.append("host_mounts")
    if any(row["Type"] not in {"bind", "tmpfs"} or row["Destination"] not in {*expected_binds, "/tmp", ROOT + "/cache"}
           for row in document["mounts"]):
        issues.append("unexpected_mounts")
    tmpfs = host.get("Tmpfs", {})
    if set(tmpfs) != {"/tmp", ROOT + "/cache"} or any(
            not all(option in value.split(",") for option in ("rw", "nosuid", "nodev", f"size={spec['tmp_mb']}m"))
            for value in tmpfs.values()):
        issues.append("private_compile_tmpfs")
    if host.get("PortBindings") or host.get("RestartPolicy", {}).get("Name") != "no":
        issues.append("ports_or_restart")
    logging = host.get("LogConfig", {})
    if logging.get("Type") != "local" or logging.get("Config") != {"max-size": "10m", "max-file": "1"}:
        issues.append("bounded_logs")
    return issues


def read_result(path, limit_mb):
    """Read only a bounded regular file after the container has stopped."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit_mb * 1024**2:
            raise ValueError("Verifier result is not a bounded regular file")
        data = handle.read(limit_mb * 1024**2 + 1)
    if len(data) > limit_mb * 1024**2:
        raise ValueError("Verifier report grew beyond its limit")
    return json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite verifier JSON")))


def proc_observation(*, root=Path("/")):
    """Run before untrusted source import; a candidate self-report is not accepted."""
    proc = root / "proc"
    rows = dict(line.split(":", 1) for line in (proc / "self/status").read_text().splitlines() if ":" in line)
    mounts = []
    for line in (proc / "self/mountinfo").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 10:
            mounts.append({"target": parts[4], "options": parts[5].split(",")})
    cgroups = {name: (root / "sys/fs/cgroup" / name).read_text().strip() for name in ("memory.max", "pids.max", "cpu.max")}
    return {"uid": os.getuid(), "gid": os.getgid(), "cap_eff": rows.get("CapEff", "").strip(),
        "cap_prm": rows.get("CapPrm", "").strip(), "cap_bnd": rows.get("CapBnd", "").strip(), "cap_amb": rows.get("CapAmb", "").strip(),
        "no_new_privs": rows.get("NoNewPrivs", "").strip(), "seccomp": rows.get("Seccomp", "").strip(),
        "network_interfaces": sorted(name for _, name in socket.if_nameindex()), "mounts": mounts, "cgroups": cgroups,
        "namespace_inodes": {name: (proc / "self/ns" / name).stat().st_ino for name in ("pid", "net", "mnt", "ipc")}}


def proc_issues(document, spec):
    issues = []
    if (document.get("uid") != 65532 or document.get("gid") != 65532
            or any(document.get(key) != "0000000000000000" for key in ("cap_eff", "cap_prm", "cap_bnd", "cap_amb"))):
        issues.append("unprivileged_identity")
    if document.get("no_new_privs") != "1" or document.get("seccomp") != "2":
        issues.append("no_new_privileges_or_seccomp")
    if document.get("network_interfaces") != ["lo"]:
        issues.append("network_interfaces")
    root = next((row for row in document.get("mounts", []) if row.get("target") == "/"), {})
    if "ro" not in root.get("options", []):
        issues.append("readonly_root")
    group = document.get("cgroups", {})
    if group.get("memory.max") != str(spec["memory_mb"] * 1024**2) or group.get("pids.max") != str(spec["pids"]):
        issues.append("cgroup_memory_or_pids")
    try:
        quota, period = map(int, group.get("cpu.max", "").split())
        if quota != spec["cpus"] * period or period <= 0:
            issues.append("cgroup_cpu_quota")
    except (ValueError, TypeError):
        issues.append("cgroup_cpu_quota")
    if not all(type(document.get("namespace_inodes", {}).get(key)) is int for key in ("pid", "net", "mnt", "ipc")):
        issues.append("namespace_observation")
    return issues


def seal(result, document, payload):
    result = copy.deepcopy(result)
    document.update(protocol=PROTOCOL, evaluator_sha256=evaluator_digest(), payload_sha256=fingerprint(payload),
                    worker_result_sha256=fingerprint(normalized(result)), security_isolation_certified=False)
    document["receipt_sha256"] = fingerprint({key: value for key, value in document.items() if key != "receipt_sha256"})
    result.setdefault("metadata", {})["execution_envelope"] = document
    return result


def validate(result, payload, *, allow_simulated=False):
    doc = result.get("metadata", {}).get("execution_envelope", {})
    spec = policy(payload["eval_config"]["isolation"])
    if (doc.get("protocol") != PROTOCOL or doc.get("evaluator_sha256") != evaluator_digest()
            or doc.get("payload_sha256") != fingerprint(payload) or doc.get("policy") != spec
            or doc.get("receipt_sha256") != fingerprint({key: value for key, value in doc.items() if key != "receipt_sha256"})
            or doc.get("worker_result_sha256") != fingerprint(normalized(result))):
        raise ValueError("Execution envelope is missing, changed or stale")
    if doc.get("simulated_execution") is not False and not allow_simulated:
        raise ValueError("Simulated isolation observations cannot establish runtime qualification")
    expected_command = create_command(spec, image=doc["image"], name=doc["command"][doc["command"].index("--name") + 1],
        device=doc["device"], request_dir=doc["request_dir"], output_dir=doc["output_dir"])
    if doc["command"] != expected_command:
        raise ValueError("Execution envelope changed the frozen container command")
    for phase in ("before", "after"):
        if inspect_issues(doc[phase], spec, image=doc["image"], device=doc["device"], request_dir=doc["request_dir"],
                          output_dir=doc["output_dir"], command=doc["command"]):
            raise ValueError("Effective Docker policy differs from the requested restrictions")
    if doc["before"]["id"] != doc["after"]["id"] or not re.fullmatch(r"[0-9a-f]{64}", doc["before"]["id"]):
        raise ValueError("Execution observations belong to different containers")
    if doc["before"]["state"].get("Running") is not False:
        raise ValueError("Container was not inspected before candidate execution")
    state = doc["after"]["state"]
    if state.get("Running") is not False or state.get("ExitCode") != 0 or state.get("OOMKilled") is not False or state.get("Error"):
        raise ValueError("Restricted container did not finish normally")
    if doc.get("cleanup_confirmed") is not True or doc.get("timed_out") is not False or doc.get("exit_code") != 0:
        raise ValueError("Restricted worker timeout/exit/cleanup is unverified")
    probe = result.get("metadata", {}).get("restricted_worker", {})
    if probe.get("payload_sha256") != fingerprint(payload) or proc_issues(probe.get("observation", {}), spec):
        raise ValueError("Worker did not observe the required Linux restrictions before candidate import")
    namespaces = probe["observation"]["namespace_inodes"]
    if any(type(doc.get("host_namespace_inodes", {}).get(key)) is not int
           or namespaces[key] == doc["host_namespace_inodes"][key] for key in ("pid", "net", "mnt", "ipc")):
        raise ValueError("Required worker namespaces are missing or shared with the parent")
    return doc


def baseline_issue(config, baseline, pid):
    from ..engine import task_payload

    if config.isolation_mode != "restricted":
        return None
    base = task_payload(config, pid, task="baseline")
    if not config.baseline_suite:
        return issue(baseline, base)
    for variant, result in baseline.get("variants", {}).items():
        if result.get("baseline_correctness") is not True:
            continue
        payload = copy.deepcopy(base)
        payload["eval_config"]["baseline_variant"] = variant
        payload["request_id"] = "kb-" + fingerprint(payload)
        if reason := issue(result, payload):
            return reason
        if result.get("metadata", {}).get("execution_envelope", {}).get("image") != baseline.get("metadata", {}).get("execution_envelope", {}).get("image"):
            return "restricted_baseline_variants_used_different_images"
    return None


def runtime_issue(reference, candidate, settings):
    if settings.get("isolation", DEFAULTS).get("mode") != "restricted":
        return None
    first = reference.get("metadata", {}).get("execution_envelope", {}).get("image")
    second = candidate.get("metadata", {}).get("execution_envelope", {}).get("image")
    if not first or first != second:
        return "restricted_evaluation_images_differ"
    return None


def issue(result, payload):
    if not enabled(payload):
        return None
    if result.get("correctness") is False or result.get("compiled") is False:
        return None
    try:
        validate(result, payload)
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return "restricted_execution_evidence_missing_stale_or_invalid"
    return None

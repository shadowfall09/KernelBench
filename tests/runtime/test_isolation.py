"""Linux/Docker policy simulation and bounded CPU file checks; never run Docker."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kb_workflows import engine
from kb_workflows.common import atomic_json, fingerprint, kernel_name
from kb_workflows.config import RunConfig, parse_config
from kb_workflows.runtime import isolation, restricted_transport

IMAGE = "sha256:" + "b" * 64
CID = "c" * 64


def observation(spec):
    return {"uid": 65532, "gid": 65532, "cap_eff": "0000000000000000", "no_new_privs": "1", "seccomp": "2",
        "cap_prm": "0000000000000000", "cap_bnd": "0000000000000000", "cap_amb": "0000000000000000",
        "network_interfaces": ["lo"], "mounts": [{"target": "/", "options": ["ro", "relatime"]}],
        "cgroups": {"memory.max": str(spec["memory_mb"] * 1024**2), "pids.max": str(spec["pids"]),
                    "cpu.max": str(spec["cpus"] * 100000) + " 100000"},
        "namespace_inodes": {name: 42 for name in ("pid", "net", "mnt", "ipc")}}


def inspected(command, *, running=False):
    image_index = command.index(IMAGE)
    def option(name):
        return command[command.index(name) + 1]
    tmpfs = dict(command[i + 1].split(":", 1) for i, value in enumerate(command) if value == "--tmpfs")
    mounts = []
    for index, value in enumerate(command):
        if value == "--mount":
            parts = command[index + 1].split(",")
            row = dict(part.split("=", 1) for part in parts if "=" in part)
            mounts.append({"Type": "bind", "Source": row["src"], "Destination": row["dst"], "RW": "readonly" not in parts, "Propagation": "rprivate"})
    return {"Id": CID, "Image": IMAGE, "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
        "CapAdd": None, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges=true"], "PidMode": "", "IpcMode": "private",
        "UTSMode": "", "UsernsMode": "", "CgroupnsMode": "private", "NanoCpus": int(option("--cpus")) * 10**9,
        "Memory": int(option("--memory")), "MemorySwap": int(option("--memory-swap")), "PidsLimit": int(option("--pids-limit")),
        "DeviceRequests": [{"DeviceIDs": [option("--gpus").split("=", 1)[1]], "Count": 0, "Driver": "", "Capabilities": [["gpu"]]}],
        "Devices": [], "Tmpfs": tmpfs, "PortBindings": {}, "LogConfig": {"Type": "local", "Config": {"max-size": "10m", "max-file": "1"}},
        "RestartPolicy": {"Name": "no"}},
        "Config": {"User": isolation.USER, "WorkingDir": isolation.ROOT, "Entrypoint": ["/usr/bin/env"],
                   "Cmd": command[image_index + 1:], "Healthcheck": {"Test": ["NONE"]},
                   "Labels": {"kernelbench-pro.owner": option("--name")}, "Env": ["SECRET_EXAMPLE=never-retain-this"]},
        "Mounts": mounts, "State": {"Running": running, "ExitCode": 0, "OOMKilled": False, "Error": ""}}


@pytest.fixture
def payload(tmp_path):
    config = RunConfig(generation="local", evaluation="docker", hardware="RTX_A6000", run_name="restricted", start=2, end=2,
                       runs_dir=str(tmp_path), isolation_mode="restricted").validate()
    config.run_dir.mkdir()
    (config.run_dir / kernel_name(1, 2)).write_text("# A simulated candidate, never imported\n")
    return engine.task_payload(config, 2)


def fake_docker(payload, *, damage=None):
    calls, state = [], {}
    def run(command, *, timeout):
        calls.append(command)
        assert timeout > 0
        action = command[1]
        if action == "image":
            return {"exit_code": 0, "output": IMAGE + "\n"}
        if action == "create":
            state["command"] = command
            return {"exit_code": 0, "output": CID + "\n"}
        if action == "inspect":
            data = inspected(state["command"])
            if damage == "network":
                data["HostConfig"]["NetworkMode"] = "host"
            if damage == "device":
                data["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["all"]
            if damage == "extra_mount":
                data["Mounts"].append({"Type": "bind", "Source": "/var/run/docker.sock", "Destination": "/socket", "RW": True})
            return {"exit_code": 0, "output": json.dumps([data])}
        if action == "start":
            command = state["command"]
            request_dir = Path(next(command[i + 1] for i, part in enumerate(command) if part == "--mount" and "dst=/request" in command[i + 1]).split("src=", 1)[1].split(",")[0])
            output_dir = Path(next(command[i + 1] for i, part in enumerate(command) if part == "--mount" and "dst=/output" in command[i + 1]).split("src=", 1)[1].split(",")[0])
            assert json.loads((request_dir / "request.json").read_text()) == payload
            probe = observation(payload["eval_config"]["isolation"])
            if damage == "probe":
                probe["seccomp"] = "0"
            raw = {"compiled": True, "correctness": True, "runtime": 1, "metadata": {
                "reference_sha256": fingerprint({"source": payload["ref_arch_src"]}), "kernel_sha256": fingerprint({"source": payload["kernel_src"]}),
                "restricted_worker": {"observation": probe, "payload_sha256": fingerprint(payload)}}}
            atomic_json(output_dir / "result.json", raw)
        if action == "wait":
            if damage == "timeout":
                raise TimeoutError("Simulated worker timeout")
            return {"exit_code": 0, "output": "0\n"}
        if action == "container":
            return {"exit_code": 0, "output": CID if damage == "cleanup" else ""}
        return {"exit_code": 0, "output": "simulated log" if action == "logs" else ""}
    return run, calls


def test_restricted_lifecycle_pins_image_uses_fresh_mounts_checks_effective_policy_and_cleans_up(payload, tmp_path):
    run, calls = fake_docker(payload)
    result = restricted_transport.evaluate(payload, SimpleNamespace(docker_image="mutable-tag"), "GPU-fixture", tmp_path / "worker.log", run=run)
    assert result["correctness"] and result["runtime"] == 1
    doc = isolation.validate(result, payload, allow_simulated=True)
    assert doc["cleanup_confirmed"] and doc["image"] == IMAGE
    assert isolation.issue(result, payload) == "restricted_execution_evidence_missing_stale_or_invalid"
    command = next(row for row in calls if row[1] == "create")
    assert "--network" in command and "--read-only" in command and "--cap-drop" in command
    assert "mutable-tag" not in command and "--privileged" not in command and "--rm" not in command
    assert "SECRET_EXAMPLE" not in json.dumps(doc)
    assert [row[1] for row in calls] == ["image", "create", "inspect", "start", "wait", "inspect", "logs", "rm", "container"]
    assert not Path(doc["request_dir"]).exists() and not Path(doc["output_dir"]).exists()


@pytest.mark.parametrize("damage", ["network", "device", "extra_mount", "probe", "timeout", "cleanup"])
def test_unenforced_or_incomplete_restrictions_never_qualify(payload, tmp_path, damage):
    run, calls = fake_docker(payload, damage=damage)
    result = restricted_transport.evaluate(payload, SimpleNamespace(docker_image="fixture"), "0", tmp_path / "worker.log", run=run)
    assert result["compiled"] is False and result["correctness"] is False
    assert any(row[1] == "rm" for row in calls)
    if damage in {"network", "device", "extra_mount"}:
        assert not any(row[1] == "start" for row in calls)


@pytest.mark.parametrize("field,bad", [("uid", 0), ("cap_eff", "0000000000000001"), ("seccomp", "0"),
                                      ("cap_prm", "0000000000000001"), ("cap_bnd", "0000000000000001"),
                                      ("cap_amb", "0000000000000001"), ("no_new_privs", "0"), ("network_interfaces", ["lo", "eth0"]), ("cgroups", {})])
def test_linux_observations_fail_closed_for_missing_or_weakened_limits(payload, field, bad):
    spec = payload["eval_config"]["isolation"]
    current = observation(spec)
    assert isolation.proc_issues(current, spec) == []
    current[field] = bad
    assert isolation.proc_issues(current, spec)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "oversized", "nonfinite"])
def test_parent_report_reader_rejects_untrusted_file_types_and_unbounded_json(tmp_path, kind):
    path = tmp_path / "result.json"
    if kind == "symlink":
        secret = tmp_path / "private.txt"
        secret.write_text('"private"')
        path.symlink_to(secret)
    elif kind == "fifo":
        import os
        os.mkfifo(path)
    elif kind == "oversized":
        path.write_bytes(b"x" * (1024**2 + 1))
    else:
        path.write_text('{"value":NaN}')
    with pytest.raises((ValueError, OSError)):
        isolation.read_result(path, 1)


def test_restricted_configuration_rejects_unsupported_transports_and_invalid_resource_claims():
    assert parse_config(["--isolation-mode", "restricted", "--evaluation", "docker", "--generation", "local"]).isolation_mode == "restricted"
    for option in ({"evaluation": "local"}, {"evaluation": "queue"}, {"isolation_cpus": True}, {"isolation_report_mb": 257}):
        with pytest.raises(ValueError):
            RunConfig(isolation_mode="restricted", **{"evaluation": "docker", **option}).validate()


def test_receipt_detects_policy_and_result_tampering_after_parent_wrappers(payload, tmp_path):
    run, _ = fake_docker(payload)
    result = restricted_transport.evaluate(payload, SimpleNamespace(docker_image="fixture"), "0", tmp_path / "worker.log", run=run)
    result["sample_id"] = 0
    result["metadata"].update(workflow_signature=fingerprint(payload), evaluation_config=payload["eval_config"])
    isolation.validate(result, payload, allow_simulated=True)
    changed = copy.deepcopy(result)
    changed["runtime"] = 0.001
    with pytest.raises(ValueError):
        isolation.validate(changed, payload, allow_simulated=True)
    changed = copy.deepcopy(result)
    changed["metadata"]["execution_envelope"]["after"]["host"]["ReadonlyRootfs"] = False
    with pytest.raises(ValueError):
        isolation.validate(changed, payload, allow_simulated=True)


def test_harbor_restricted_export_uses_real_separate_bundled_verifier_schema(tmp_path):
    import tomllib

    import yaml
    Task = pytest.importorskip("harbor.models.task.task").Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    from kb_workflows.common import read_json
    from kb_workflows.runtime import harbor

    config = RunConfig(generation="harbor", evaluation="docker", stage="export-harbor", hardware="RTX_A6000", start=2, end=2,
                       isolation_mode="restricted", isolation_cpus=3, isolation_memory_mb=32768).validate()
    path = harbor.write_task(tmp_path / "task", config, 2)
    task = Task(path)
    definition = resolve_verifier_environment_definition(task.config, task.paths)
    assert definition is not None and definition.bundled_tests
    parsed = tomllib.loads((path / "task.toml").read_text())
    assert parsed["verifier"]["network_mode"] == "no-network"
    assert parsed["verifier"]["environment"]["cpus"] == 3
    assert parsed["verifier"]["environment"]["memory_mb"] == 32768
    service = yaml.safe_load((path / "tests/docker-compose.yaml").read_text())["services"]["main"]
    assert service["read_only"] and service["user"] == "65532:65532" and service["network_mode"] == "none"
    assert service["cap_drop"] == ["ALL"] and service["pids_limit"] == config.isolation_pids
    assert any(value.startswith("/app:") for value in service["tmpfs"])
    assert not (path / "environment/docker-compose.yaml").exists()
    spec = read_json(path / "tests/config.json")
    assert spec["restricted_policy"] == isolation.options(config)
    assert spec["config"]["isolation_mode"] == "standard"


def test_harbor_agent_allowlist_is_explicit_and_validates_against_the_actual_task_schema(tmp_path):
    Task = pytest.importorskip("harbor.models.task.task").Task

    from kb_workflows.runtime import harbor

    for value in ("https://api.example.test", "api.example.test/path", "*", "host:8443", "host,,api.example.test"):
        with pytest.raises(ValueError):
            isolation.allowed_hosts(value)
    config = RunConfig(generation="harbor", evaluation="docker", hardware="RTX_A6000", isolation_mode="restricted",
                       harbor_agent_hosts="api.example.test,192.0.2.1", start=2, end=2).validate()
    task = Task(harbor.write_task(tmp_path / "allowed-task", config, 2))
    assert task.config.agent.network_mode.value == "allowlist"
    assert task.config.agent.allowed_hosts == ["api.example.test", "192.0.2.1"]
    assert task.config.verifier.network_mode.value == "no-network"
    with pytest.raises(ValueError, match="agent-hosts"):
        RunConfig(generation="harbor", evaluation="docker", hardware="RTX_A6000", isolation_mode="restricted").validate()


@pytest.mark.parametrize("bad_probe", [False, True])
def test_restricted_bootstrap_checks_linux_before_dispatch_and_makes_a_bounded_readable_handoff(payload, tmp_path, monkeypatch, bad_probe):
    import os
    import sys

    from kb_workflows.common import read_json
    from kb_workflows.evaluation import evaluate_task
    from kb_workflows.runtime import isolated_worker

    source, target = tmp_path / "request.json", tmp_path / "result.json"
    atomic_json(source, payload)
    probe = observation(payload["eval_config"]["isolation"])
    if bad_probe:
        probe["seccomp"] = "0"
    monkeypatch.setattr(isolated_worker, "proc_observation", lambda: probe)
    seen = []
    def dispatch():
        seen.append(True)
        atomic_json(target, {"compiled": True, "correctness": True, "runtime": 1, "metadata": {}})
    monkeypatch.setattr(evaluate_task, "main", dispatch)
    monkeypatch.setattr(sys, "argv", ["isolated_worker", str(source), str(target)])
    isolated_worker.main()
    result = read_json(target)
    assert result["correctness"] is not bad_probe and bool(seen) is not bad_probe
    assert os.stat(target).st_mode & 0o777 == 0o644
    if not bad_probe:
        assert result["metadata"]["restricted_worker"]["payload_sha256"] == fingerprint(payload)


def test_a_plain_local_process_cannot_satisfy_restricted_verification(payload, tmp_path, monkeypatch):
    from kb_workflows.runtime import transports

    monkeypatch.setattr(transports.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must reject before launching a plain worker"))
    with pytest.raises(ValueError, match="local process"):
        transports.local_evaluate(payload, "0", tmp_path / "unrestricted.log")


def test_restricted_scores_cannot_mix_runtime_images_even_with_the_same_torch_version(payload):
    def result(image):
        return {"metadata": {"execution_envelope": {"image": image}}}
    settings = payload["eval_config"]
    assert isolation.runtime_issue(result(IMAGE), result(IMAGE), settings) is None
    assert isolation.runtime_issue(result(IMAGE), result("sha256:" + "d" * 64), settings)
    assert isolation.runtime_issue({}, {}, settings)
    assert isolation.runtime_issue({}, {}, {"isolation": isolation.DEFAULTS}) is None

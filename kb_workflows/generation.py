"""Generation shared by local processes, Docker, Modal and Harbor."""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

from .common import ROOT, atomic_json, evaluator_digest, fingerprint, kernel_name, read_json, select_problems
from .reference_population import select_run_problems
from .transports import resolve_devices, terminate


def input_blind_problems():
    tree = ast.parse((ROOT / "src" / "prompt_constructor.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "STRIP_TEST_CONFIG_PIDS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Input-blind problem configuration is missing")


def prompt_source(source, level, pid):
    # Follow the configured input-blind policy without importing its API-client dependencies.
    if level == 5:
        lines = source.splitlines(keepends=True)
        return "\n".join("".join(lines[node.lineno - 1:node.end_lineno]) for node in ast.parse(source).body
                         if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         or node.name not in {"get_inputs", "get_init_inputs"})
    if (level, pid) not in input_blind_problems():
        return source
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    return "\n".join(
        "".join(lines[node.lineno - 1 : node.end_lineno])
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
    )


def agent_environment(config):
    environment = {}
    if config.bedrock:
        environment["CLAUDE_CODE_USE_BEDROCK"] = "1"
    if config.profile:
        environment["AWS_PROFILE"] = config.profile
    if config.model:
        model = config.model
        if config.bedrock:
            model = {"sonnet": "global.anthropic.claude-sonnet-4-6", "opus": "global.anthropic.claude-opus-4-6-v1"}.get(
                model, model
            )
        environment["ANTHROPIC_MODEL"] = model
    if config.small_model:
        environment["ANTHROPIC_SMALL_FAST_MODEL"] = config.small_model
    return environment


def build_prompt(config, pid):
    filename = kernel_name(config.level, pid)
    if config.state_validation:
        from .state_policies import plan as state_plan

        state_plan(config.level, pid, select_problems(config.level, problem_ids=[pid])[pid].read_text())
    if config.cache_validation:
        from .cache_recipes import plan as cache_plan

        cache_plan(config.level, pid, select_problems(config.level, problem_ids=[pid])[pid].read_text())
    if config.level in {1, 2, 3}:
        from .operator_input_policy import plan

        plan(config.level, pid, select_problems(config.level, problem_ids=[pid])[pid].read_text(), config.input_family_memory_mb)
    contract = ""
    if config.level == 5:
        from .production_specs import task

        spec = task(pid)
        contract = (f"\nTrack: {spec['track']}; activation family: {spec['fixed_activation_dtype'] or config.precision}. "
                    "Input dtypes follow the public per-role contract; keep scale factors FP32 and indices/masks discrete. "
                    "Packed FP4 references specify software E2M1 semantics, not mandatory native FP4 instructions.\n")
    if config.input_calibration:
        from .input_calibration import record_for

        source = select_problems(config.level, problem_ids=[pid])[pid].read_text()
        row = record_for(config.input_calibration, config.level, pid, config.precision, source)
        contract += (f"\nNominal value distribution: {row['selected_recipe']}; shapes and discrete roles stay unchanged. "
                     f"Frozen RMS/all-element FP64 error policy: {row['precision_contract']}. "
                     "Both the reference and submission must qualify. Hidden distributions remain separate.\n")
    implementation = ("real NKI kernels using import nki and native torch_neuronx.nki_op registration, taking neuron tensors; keep the ModelNew torch.nn.Module interface"
                      if config.device_backend == "neuron" else "real CUDA/Triton kernels")
    return f"""Implement ModelNew for KernelBench Pro level {config.level}, problem {pid}.
The public reference is supplied below; implement {implementation}.
Target hardware: {config.hardware or config.gpu}; architecture: {config.gpu_arch}; requested precision family: {config.precision}.{contract}
Return one complete Python source block. The controller writes it to runs/agent/{filename}.
Keep the model's algorithm, input-dependent behavior, shapes and parameters correct.
Some references intentionally omit the test harness. Implement from the supplied Model definition;
do not inspect hidden_tests, original test inputs, or other repository problem files.
Mode: {config.generation_mode}. Candidate budget: {1 if config.generation_mode == 'one-shot' else config.optimization_rounds}.
Wall-clock budget: {config.agent_timeout} seconds for this sample, including development evaluation.
{"No execution or test feedback is provided before the final submission." if config.generation_mode == 'one-shot' else "After each candidate the controller returns standard correctness, timing and optional profiler diagnostics. Improve using that feedback."}
The controller retains the fastest development-correct candidate. The hidden input distributions are evaluated only after final selection.
Interactive development entrypoint: python verify.py (standard-only feedback; unavailable in one-shot mode).
Profiler timings are diagnostic, not final benchmark timings. Do not change the verifier or reference source.
"""


def prepare_job(config, pid, sid):
    from .reference_population import freeze_run

    population = freeze_run(config)
    if pid not in select_run_problems(config):
        raise ValueError("Agent task is outside the declared reference population")
    directory = config.run_dir / "work" / f"p{pid}_s{sid}"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in (directory / "agent_result.json", directory / "best_candidate.json", directory / "development_history.json",
                  directory / "trajectory.json", directory / "interactive_state.json", directory / "claude.stderr.log",
                  directory / "runs" / "agent" / kernel_name(config.level, pid)):
        stale.unlink(missing_ok=True)
    for name in ("rounds", "model_calls"):
        path = directory / name
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    source = select_problems(config.level, problem_ids=[pid])[pid].read_text()
    (directory / "model.py").write_text(prompt_source(source, config.level, pid))
    job = config.to_dict()
    job.update(
        suite="production41" if config.level == 5 else "full",
        suite_role="scored",
        start=pid,
        end=pid,
        num_samples=1,
        run_name="agent",
        resume=False,
        stage="evaluate",
        runs_dir=str(directory / "runs"),
        build_image=False,
        record_baseline=False,
        problem_specs="",  # Layer specifications are used only by parent analysis.
        score_mode="bounded_gain",
        device_calibration="",
        reference_population="",
        reference_population_scope="run",
    )
    # Evaluation inside a GPU container/sandbox uses its allocated device 0.
    if config.generation == "docker" and config.evaluation == "docker":
        job.update(evaluation="local", gpus="0")
    if config.generation == "docker" and config.evaluation == "local":
        job.update(gpus="0")
    if config.generation == "modal":
        job.update(evaluation="local", gpus="0")
    if config.generation == "docker":
        job["runs_dir"] = "/work/runs"
        if config.evaluation == "queue" and "127.0.0.1" in job["queue_url"]:
            job["queue_url"] = job["queue_url"].replace("127.0.0.1", "host.docker.internal")
        if config.evaluation == "queue" and "localhost" in job["queue_url"]:
            job["queue_url"] = job["queue_url"].replace("localhost", "host.docker.internal")
    if config.generation == "modal":
        job["runs_dir"] = "/work/runs"
    if config.input_calibration:
        from .input_calibration import subset

        atomic_json(directory / "input_calibration.json", subset(config.input_calibration, config.level, pid, config.precision, source, population=population))
        job["input_calibration"] = ("/work/input_calibration.json" if config.generation in {"docker", "modal"}
                                    else str(directory / "input_calibration.json"))
    if config.determinism_policy_file:
        from .determinism import POLICY_PROTOCOL, read_policies

        # Only the selected public numerical contract enters the development job.
        key = f"{config.level}/{pid}"
        policies = read_policies(config.determinism_policy_file)["tasks"]
        atomic_json(directory / "determinism_policy.json", {"protocol": POLICY_PROTOCOL,
                    "tasks": {key: policies[key]} if key in policies else {}})
        job["determinism_policy_file"] = ("/work/determinism_policy.json" if config.generation in {"docker", "modal"}
                                          else str(directory / "determinism_policy.json"))
    atomic_json(directory / "job_config.json", job)
    (directory / "verify.py").write_text(
        "from pathlib import Path\nfrom kb_workflows.agent_task import verify_directory\n"
        "raise SystemExit(verify_directory(Path(__file__).resolve().parent))\n"
    )
    return directory


def run_agent(directory, config, pid, device=None):
    output = directory / "runs" / "agent" / kernel_name(config.level, pid)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A non-resume retry must not accidentally report an old file as new output.
    output.unlink(missing_ok=True)
    with (directory / "agent.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "kb_workflows.agent_loop", "--directory", str(directory)],
            cwd=directory,
            env={
                **os.environ,
                **agent_environment(config),
                "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "MSWEA_SILENT_STARTUP": "1",
                **(({"NEURON_RT_VISIBLE_CORES": str(device), "NEURON_LOGICAL_NC_CONFIG": "1"} if config.device_backend == "neuron"
                    else {"CUDA_VISIBLE_DEVICES": str(device)}) if device is not None else {}),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=config.agent_timeout)
        except subprocess.TimeoutExpired:
            terminate_agent_tree(process)
            best = read_json(directory / "best_candidate.json", {})
            if best.get("source") and (best.get("development_verified") or config.generation_mode == "one-shot"):
                output.write_text(best["source"])
            else:
                output.unlink(missing_ok=True)
            return {"status": "success" if output.exists() else "timeout", "exit_status": "WallTimeExceeded",
                    "error": "Agent wall-clock budget exhausted; retained any completed best candidate",
                    "budget": None, "resource_receipt_incomplete": True}
        finally:
            terminate(process)
    return read_json(directory / "agent_result.json", {"status": "failed", "exit_code": process.returncode})


def terminate_agent_tree(process):
    """Stop this controller's nested model/evaluation process groups on wall timeout."""
    import psutil

    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    terminate(process)
    psutil.wait_procs(children, timeout=2)


def run_docker_agent(directory, config, pid, device):
    name = "kb-agent-" + uuid.uuid4().hex[:12]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--add-host=host.docker.internal:host-gateway",
        "-v",
        f"{directory}:/work",
        "-w",
        "/work",
    ]
    if config.evaluation in {"local", "docker"}:
        command += ["--gpus", f"device={device}"]
    if config.enable_ncu and config.evaluation in {"local", "docker"}:
        command += ["--cap-add=SYS_ADMIN", "--security-opt", "seccomp=unconfined"]
    variables = {
        **{
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "GEMINI_API_KEY",
                "CLAUDE_CODE_USE_BEDROCK",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "AWS_REGION",
                "AWS_DEFAULT_REGION",
                "MODAL_TOKEN_ID",
                "MODAL_TOKEN_SECRET",
            }
        },
        **agent_environment(config),
    }
    # Values are inherited by Docker rather than copied into command arguments/logs.
    for key in variables:
        command += ["-e", key]
    aws = Path.home() / ".aws"
    if config.bedrock and aws.exists():
        command += ["-v", f"{aws}:/root/.aws:ro"]
    modal_auth = Path.home() / ".modal.toml"
    if config.evaluation == "modal" and modal_auth.exists():
        command += ["-v", f"{modal_auth}:/root/.modal.toml:ro"]
    command += [
        config.docker_image,
        "python",
        "/app/KernelBench/scripts/run_agent.py",
        "--config",
        "/work/job_config.json",
        "--directory",
        "/work",
    ]
    with (directory / "container.log").open("w") as log:
        process = subprocess.Popen(
            command, env={**os.environ, **variables}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            process.wait(timeout=config.agent_timeout + 60)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Generation container timed out"}
        finally:
            terminate(process)
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return read_json(directory / "agent_result.json", {"status": "failed", "exit_code": process.returncode})


def build_docker_image(config):
    target = "agent" if config.evaluation in {"local", "docker"} else "agent-remote"
    subprocess.run(["docker", "build", "--target", target, "-t", config.docker_image, str(ROOT)], check=True)


def generation_signature(config, pid, sid):
    from .experiment import protocol
    from .local_model import checkpoint_digest

    return fingerprint(
        {
            "reference": select_problems(config.level, problem_ids=[pid])[pid].read_text(),
            "prompt": build_prompt(config, pid),
            "sample_id": sid,
            "generation": config.generation,
            "harbor": {"agent": config.harbor_agent, "environment": config.harbor_environment,
                       "command": config.harbor_command, "reward": config.harbor_reward,
                       "evaluator_sha256": evaluator_digest()} if config.generation == "harbor" else None,
            "model": config.model,
            "local_checkpoint_sha256": checkpoint_digest(config.model) if config.agent_backend == "local-transformers" else None,
            "small_model": config.small_model,
            "bedrock": config.bedrock,
            "precision": config.precision,
            "production_instance": {"profile": config.task_profile, "memory_mb": config.task_memory_mb,
                                    "dimensions": read_json(Path(config.task_dimensions)) if config.task_dimensions else None}
                if config.level == 5 else None,
            "experiment": protocol(config),
            "evaluator_sha256": evaluator_digest(),
        }
    )


def generate_run(config):
    from .reference_population import freeze_run
    from .resource_budgets import active

    freeze_run(config)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    target = config.run_dir / "generation_summary.json"
    previous = read_json(target, {})
    summary = previous if config.resume else {}
    history_path = config.run_dir / "generation_attempts.json"
    history = read_json(history_path, {"slots": {}, "historical_attempts_unknown": bool(previous)})
    tasks = []
    for pid in select_run_problems(config):
        for sid in range(config.num_samples):
            key = f"{pid}_{sid}"
            path = config.run_dir / kernel_name(config.level, pid, sid)
            entry = summary.get(key, {})
            if (
                config.resume
                and entry.get("signature") == generation_signature(config, pid, sid)
                and entry.get("status") in {"success", "failed", "timeout", "error"}
                and entry.get("kernel_sha256") == (hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
            ):
                print(f"[generate] {key}: matching completed attempt ({entry['status']})", flush=True)
                continue
            if active(config) and (history.get("slots", {}).get(key) or key in previous):
                raise ValueError(f"Capped sample {key} already has an attempt; use resume for identical completed work or a new run name")
            tasks.append((pid, sid))
    local_gpu = ((config.generation in {"local", "docker"} and config.evaluation in {"local", "docker"})
                 or (config.generation == "harbor" and config.harbor_environment == "docker"))
    devices = resolve_devices(config.gpus, backend=config.device_backend) if local_gpu else ["0"]
    # Generation with a local GPU must not run two agents concurrently on that GPU.
    import threading

    locks = [threading.Lock() for _ in devices]
    workers = min(config.parallel, len(devices)) if local_gpu else config.parallel

    def run(index, pid, sid):
        directory = prepare_job(config, pid, sid)
        started = time.monotonic()
        try:
            lane = index % len(devices)
            with locks[lane] if local_gpu else nullcontext():
                if config.generation == "harbor":
                    from .harbor import run_harbor_agent

                    result = run_harbor_agent(directory, config, pid, devices[lane] if local_gpu else None, sid=sid)
                elif config.generation == "modal":
                    from .modal_backend import run_modal_agent

                    result = run_modal_agent(directory, config, pid)
                elif config.generation == "docker":
                    result = run_docker_agent(directory, config, pid, devices[lane])
                else:
                    # Set the verifier's physical device in its per-job configuration.
                    job = read_json(directory / "job_config.json")
                    job["gpus"] = "0" if local_gpu else config.gpus
                    atomic_json(directory / "job_config.json", job)
                    result = run_agent(directory, config, pid, devices[lane] if local_gpu else None)
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        result.update(elapsed=time.monotonic() - started, signature=generation_signature(config, pid, sid))
        source = directory / "runs" / "agent" / kernel_name(config.level, pid)
        destination = config.run_dir / kernel_name(config.level, pid, sid)
        if source.exists() and source.stat().st_size:
            destination.write_bytes(source.read_bytes())
            result["kernel_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        else:
            # Prevent the final evaluator from using a stale kernel after a failed overwrite.
            destination.unlink(missing_ok=True)
        return f"{pid}_{sid}", result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid, sid in tasks:
            history["slots"].setdefault(f"{pid}_{sid}", []).append({"status": "started", "budget": None,
                                                                 "started_at": time.time(), "signature": generation_signature(config, pid, sid)})
        atomic_json(history_path, history)
        futures = [pool.submit(run, index, pid, sid) for index, (pid, sid) in enumerate(tasks)]
        for future in as_completed(futures):
            key, result = future.result()
            summary[key] = result
            history["slots"][key][-1].update(status=result["status"], budget=result.get("budget"), finished_at=time.time())
            atomic_json(history_path, history)
            atomic_json(target, summary)
            print(f"[generate] {key}: {result['status']}", flush=True)
    atomic_json(target, summary)
    from .experiment import write_budget_report

    write_budget_report(config, summary)
    return all(
        summary.get(f"{pid}_{sid}", {}).get("status") == "success"
        for pid in select_run_problems(config)
        for sid in range(config.num_samples)
    )

from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import ROOT, atomic_json, evaluator_digest, failure, fingerprint, kernel_name, read_json, select_problems
from .transports import docker_evaluate, local_evaluate, queue_evaluate, resolve_devices


def settings(config, hidden=False):
    # Changes to the actual evaluator invalidate resumed evaluations.
    return {
        "precision": config.precision,
        "fp32_tolerance": config.fp32_tolerance,
        "num_correct_trials": config.num_correct_trials,
        "num_perf_trials": config.num_perf_trials,
        "timeout": config.timeout,
        "gpu_arch": config.gpu_arch,
        "gpu": config.gpu,
        "hardware": config.hardware,
        "evaluation_backend": config.evaluation,
        "use_hidden_tests": hidden,
        "measure_performance": not hidden,
        "evaluator_sha256": evaluator_digest(),
    }


def task_payload(config, pid, sid=0, hidden=False, task="evaluate"):
    path = select_problems(config.level, problem_ids=[pid])[pid]
    options = settings(config, hidden)
    payload = {
        "task": task,
        "level": config.level,
        "problem_id": pid,
        "sample_id": sid,
        "ref_arch_src": path.read_text(),
        "eval_config": options,
    }
    if task == "evaluate":
        payload["kernel_src"] = (config.run_dir / kernel_name(config.level, pid, sid)).read_text()
    else:
        options.update(compile_backend=config.compile_backend, compile_mode=config.compile_mode)
    if hidden:
        hidden_path = ROOT / "hidden_tests" / f"level{config.level}" / f"{pid}_hidden.py"
        if not hidden_path.is_file():
            raise FileNotFoundError(f"Hidden tests missing: {hidden_path}")
        options["hidden_source_sha256"] = hashlib.sha256(hidden_path.read_bytes()).hexdigest()
    payload["request_id"] = "kb-" + fingerprint(payload)
    return payload


def evaluate_tasks(config, payloads):
    """Keep each local GPU serial while allowing other GPUs/transports to progress."""
    if not payloads:
        return
    devices = resolve_devices(config.gpus) if config.evaluation in {"local", "docker"} else ["0"]
    locks = [threading.Lock() for _ in devices]
    workers = (
        len(devices)
        if config.evaluation in {"local", "docker"}
        else (config.queue_max_inflight if config.evaluation == "queue" else config.parallel)
    )

    def run(index, key, payload):
        try:
            if config.evaluation in {"local", "docker"}:
                lane = index % len(devices)
                with locks[lane]:
                    log = config.run_dir / "logs" / f"{key}.log"
                    result = (
                        local_evaluate(payload, devices[lane], log)
                        if config.evaluation == "local"
                        else docker_evaluate(payload, config, devices[lane], log)
                    )
            elif config.evaluation == "queue":
                result = queue_evaluate(payload, config)
            else:
                from .modal_backend import modal_evaluate

                result = modal_evaluate(payload, config)
        except Exception as exc:
            result = failure(str(exc), error_name=type(exc).__name__)
        result.setdefault("metadata", {})["workflow_signature"] = fingerprint(payload)
        result["metadata"]["evaluation_config"] = payload["eval_config"]
        return key, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, index, key, payload) for index, (key, payload) in enumerate(payloads)]
        for future in as_completed(futures):
            yield future.result()


def evaluate_run(config):
    selected = select_problems(config.level, config.start, config.end)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for hidden in [False, True] if config.hidden_tests else [False]:
        label = "hidden" if hidden else "standard"
        target = config.run_dir / ("eval_results_hidden.json" if hidden else "eval_results.json")
        results = read_json(target, {}) if config.resume else {}
        tasks = []
        for pid in selected:
            for sid in range(config.num_samples):
                key = f"{pid}_{sid}"
                path = config.run_dir / kernel_name(config.level, pid, sid)
                if not path.is_file() or not path.stat().st_size:
                    results[key] = {"sample_id": sid, **failure("Kernel file missing")}
                    continue
                payload = task_payload(config, pid, sid, hidden)
                signature = fingerprint(payload)
                existing = results.get(key, {})
                if config.resume and existing.get("metadata", {}).get("workflow_signature") == signature:
                    print(f"[{label}] {key}: matching cached result", flush=True)
                    continue
                tasks.append((f"{label}-{key}", payload))
        for tagged_key, result in evaluate_tasks(config, tasks):
            key = tagged_key.removeprefix(f"{label}-")
            result["sample_id"] = int(key.rsplit("_", 1)[1])
            results[key] = result
            atomic_json(target, results)
            print(f"[{label}] {key}: compiled={result['compiled']}, correct={result['correctness']}", flush=True)
        atomic_json(target, results)
    write_pass_at_k(config)
    return config.run_dir


def write_pass_at_k(config):
    standard = read_json(config.run_dir / "eval_results.json", {})
    hidden = read_json(config.run_dir / "eval_results_hidden.json", {}) if config.hidden_tests else standard
    n = config.num_samples
    ks = sorted({int(k) for k in config.pass_at_k.split(",") if int(k) <= n})
    problems = {}
    for pid in select_problems(config.level, config.start, config.end):
        count = sum(
            bool(standard.get(f"{pid}_{sid}", {}).get("compiled"))
            and bool(standard.get(f"{pid}_{sid}", {}).get("correctness"))
            and bool(hidden.get(f"{pid}_{sid}", {}).get("correctness"))
            for sid in range(n)
        )
        problems[str(pid)] = {
            "total_samples": n,
            "correct_samples": count,
            **{f"pass@{k}": 1 - math.comb(n - count, k) / math.comb(n, k) if n - count >= k else 1.0 for k in ks},
        }
    atomic_json(
        config.run_dir / "pass_at_k_results.json",
        {
            "hidden_gated": config.hidden_tests,
            "problems": problems,
            "averages": {f"avg_pass@{k}": sum(p[f"pass@{k}"] for p in problems.values()) / len(problems) for k in ks},
        },
    )


def record_baseline(config):
    if not config.hardware:
        raise ValueError("Specify --hardware when recording a baseline (use the actual verifier GPU name)")
    target = ROOT / "results" / "timing" / config.hardware / f"{config.baseline}.json"
    results = read_json(target, {})
    level = results.setdefault(f"level{config.level}", {})
    names = {}
    tasks = []
    for pid, path in select_problems(config.level, config.start, config.end).items():
        payload = task_payload(config, pid, task="baseline")
        signature = fingerprint(payload)
        if config.resume and (level.get(path.name) or {}).get("metadata", {}).get("workflow_signature") == signature:
            continue
        key = f"baseline-{pid}"
        names[key] = path.name
        tasks.append((key, payload))
    for key, result in evaluate_tasks(config, tasks):
        level[names[key]] = result
        atomic_json(target, results)
        print(f"[baseline] {names[key]}: {result.get('mean', result.get('metadata'))}", flush=True)
    atomic_json(target, results)
    return target


def analyze_run(config):
    if not config.hardware:
        raise ValueError("Specify --hardware to choose the matching baseline")
    from scripts.benchmark_eval_analysis import analyze_multi_sample_eval

    selected = select_problems(config.level, config.start, config.end)
    return analyze_multi_sample_eval(
        config.run_name,
        config.hardware,
        config.baseline,
        config.level,
        use_hidden_eval=config.hidden_tests,
        problem_ids=list(selected),
        runs_dir=config.runs_dir,
        detail_limit=30,
    )

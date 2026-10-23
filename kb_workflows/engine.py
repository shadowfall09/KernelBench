from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import ROOT, atomic_json, evaluator_digest, failure, fingerprint, kernel_name, read_json, select_problems
from .suites import selection_receipt
from .tool_checks import diagnostic_options, reusable
from .transports import docker_evaluate, local_evaluate, queue_evaluate, resolve_devices


def settings(config, hidden=False):
    from .isolation import options

    # Changes to the actual evaluator invalidate resumed evaluations.
    return {
        "precision": config.precision,
        "fp32_tolerance": config.fp32_tolerance,
        "num_correct_trials": config.num_correct_trials,
        "num_perf_trials": config.num_perf_trials,
        "timeout": config.timeout,
        "gpu_arch": config.gpu_arch,
        "device_backend": config.device_backend,
        "gpu": config.gpu,
        "hardware": config.hardware,
        "evaluation_backend": config.evaluation,
        "use_hidden_tests": hidden,
        "measure_performance": not hidden,
        "timing_method": "synchronized_wall",
        "baseline_suite": config.baseline_suite,
        "baseline_portfolio": config.baseline_portfolio,
        "paired_timing": config.paired_timing,
        "paired_trials": config.paired_trials,
        "final_repeats": config.final_repeats,
        "cold_calls": config.cold_calls,
        "score_gate_required": config.score_gate_required,
        "strict_validation": config.strict_validation,
        "evidence_level": config.evidence_level,
        "evaluator_sha256": evaluator_digest(),
        **diagnostic_options(config),
        "isolation": options(config),
    }


def task_payload(config, pid, sid=0, hidden=False, task="evaluate"):
    path = select_problems(config.level, problem_ids=[pid])[pid]
    options = settings(config, hidden)
    from .reference_revisions import binding

    if revision := binding(config.level, pid, path.read_text()):
        options["reference_revision"] = revision
    from .operator_input_policy import plan

    if config.level in {1, 2, 3}:
        options["operator_inputs"] = plan(config.level, pid, path.read_text(), config.input_family_memory_mb)
    if config.input_calibration:
        from .input_calibration import record_for

        options["input_profile"] = record_for(config.input_calibration, config.level, pid, config.precision, path.read_text())
    if config.numerical_validation:
        from .operator_specs import spec_for

        options["oracle_spec_sha256"] = fingerprint(spec_for(config.level, pid, path.read_text()))
    if config.cache_validation:
        from .cache_recipes import plan as cache_plan

        options["input_cache_plan"] = cache_plan(config.level, pid, path.read_text())
    if config.determinism_checks:
        from .determinism import policy_for

        options["determinism_policy"] = policy_for(config.determinism_policy_file, config.level, pid, path.read_text())
    if config.level == 5:
        from .production import PROTOCOL
        from .production_specs import task as production_task

        spec = production_task(pid)
        options["precision"] = spec["fixed_activation_dtype"] or config.precision
        options["production"] = {"protocol": PROTOCOL, "task_sha256": fingerprint(spec),
                                 "profile": config.task_profile, "memory_bytes": config.task_memory_mb * 1024**2}
        if config.task_dimensions:
            from .production import instance_plan, read_dimensions

            options["production"]["dimensions_override"] = read_dimensions(config.task_dimensions)
            instance_plan(pid, precision=options["precision"], profile=config.task_profile,
                          memory_bytes=config.task_memory_mb * 1024**2,
                          dimensions_override=options["production"]["dimensions_override"])
    payload = {
        "task": task,
        "level": config.level,
        "problem_id": pid,
        "sample_id": sid,
        "ref_arch_src": path.read_text(),
        "eval_config": options,
    }
    if config.device_python and config.evaluation == "local":
        payload["worker_python"] = config.device_python
    if task in {"evaluate", "profile", "paired_timing", "qualify_small", "origin_audit"}:
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
    devices = resolve_devices(config.gpus, backend=config.device_backend) if config.evaluation in {"local", "docker"} else ["0"]
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
    selected = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track)
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
                existing = results.get(key, {})
                if config.resume and reusable(existing, payload):
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
    if config.final_repeats:
        from .final_verification import run

        run(config)
    if config.paired_timing:
        run_paired(config)
    write_pass_at_k(config)
    return config.run_dir


def write_pass_at_k(config):
    from .final_verification import correctness_passed

    standard = read_json(config.run_dir / "eval_results.json", {})
    hidden = read_json(config.run_dir / "eval_results_hidden.json", {}) if config.hidden_tests else standard
    n = config.num_samples
    ks = sorted({int(k) for k in config.pass_at_k.split(",") if int(k) <= n})
    problems = {}
    for pid in select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track):
        count = sum(
            bool(standard.get(f"{pid}_{sid}", {}).get("compiled"))
            and bool(standard.get(f"{pid}_{sid}", {}).get("correctness"))
            and bool(hidden.get(f"{pid}_{sid}", {}).get("correctness"))
            and correctness_passed(config, pid, sid)
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
            "suite": selection_receipt(config),
            "hidden_gated": config.hidden_tests,
            "problems": problems,
            "averages": {f"avg_pass@{k}": sum(p[f"pass@{k}"] for p in problems.values()) / len(problems) for k in ks},
        },
    )
    write_validation_summary(config, standard, hidden)


def write_validation_summary(config, standard=None, hidden=None):
    """Keep compilation, numeric correctness and measured performance distinct."""
    standard = read_json(config.run_dir / "eval_results.json", {}) if standard is None else standard
    if hidden is None:
        hidden = read_json(config.run_dir / "eval_results_hidden.json", {}) if config.hidden_tests else standard
    counts = dict(declared_samples=0, compiled_samples=0, standard_correct_samples=0,
                  hidden_correct_samples=0, correct_samples=0, timed_correct_samples=0)
    rows = []
    for pid in select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track):
        for sid in range(config.num_samples):
            key = f"{pid}_{sid}"
            raw, secret = standard.get(key, {}), hidden.get(key, {})
            compiled = bool(raw.get("compiled"))
            standard_ok = compiled and bool(raw.get("correctness"))
            hidden_ok = bool(secret.get("compiled") and secret.get("correctness")) if config.hidden_tests else standard_ok
            correct = standard_ok and hidden_ok
            runtime = raw.get("runtime")
            timed = correct and type(runtime) in (int, float) and math.isfinite(runtime) and runtime > 0
            for field, passed in (("declared_samples", True), ("compiled_samples", compiled),
                                  ("standard_correct_samples", standard_ok), ("hidden_correct_samples", hidden_ok and config.hidden_tests),
                                  ("correct_samples", correct), ("timed_correct_samples", timed)):
                counts[field] += int(passed)
            rows.append({"problem_id": pid, "sample_id": sid, "compiled": compiled,
                         "standard_correct": standard_ok, "hidden_correct": hidden_ok if config.hidden_tests else None,
                         "correct": correct, "timed_correct": timed,
                         "numeric_stress_enabled": secret.get("metadata", {}).get("numeric_stress_enabled", False),
                         "checks": secret.get("metadata", {}).get("hidden_completed_checks", 0)})
    report = {"suite": selection_receipt(config), "hidden_tests_enabled": config.hidden_tests, "counts": counts, "samples": rows,
              "purpose": "raw_stage_diagnostics", "score_eligibility_checked": False,
              "functional_validity_certified": False}
    atomic_json(config.run_dir / "validation_summary.json", report)
    return report


def record_baseline(config):
    from .input_calibration import issue as calibrated_input_issue
    from .isolation import issue as isolation_issue

    if not config.hardware:
        raise ValueError("Specify --hardware when recording a baseline (use the actual verifier GPU name)")
    if config.baseline_suite:
        return record_baseline_suite(config)
    target = ROOT / "results" / "timing" / config.hardware / f"{config.baseline}.json"
    results = read_json(target, {})
    level = results.setdefault(f"level{config.level}", {})
    names = {}
    tasks = []
    for pid, path in select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track).items():
        payload = task_payload(config, pid, task="baseline")
        signature = fingerprint(payload)
        if (config.resume and (level.get(path.name) or {}).get("metadata", {}).get("workflow_signature") == signature
                and not isolation_issue(level[path.name], payload) and not calibrated_input_issue(level[path.name], payload)):
            continue
        key = f"baseline-{pid}"
        names[key] = path.name
        tasks.append((key, payload))
    for key, result in evaluate_tasks(config, tasks):
        expected = next(payload for task_key, payload in tasks if task_key == key)
        if reason := calibrated_input_issue(result, expected):
            result = failure(reason)
        level[names[key]] = result
        atomic_json(target, results)
        print(f"[baseline] {names[key]}: {result.get('mean', result.get('metadata'))}", flush=True)
    atomic_json(target, results)
    return target


def analyze_run(config):
    if not config.hardware:
        raise ValueError("Specify --hardware to choose the matching baseline")
    from scripts.benchmark_eval_analysis import analyze_multi_sample_eval

    from .paper_analysis import analyze_paper

    write_validation_summary(config)
    report = analyze_paper(config)
    try:
        analyze_multi_sample_eval(
            config.run_name, config.hardware, config.baseline, config.level,
            use_hidden_eval=config.hidden_tests,
            problem_ids=list(select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track)),
            runs_dir=config.runs_dir, detail_limit=30,
        )
    except (ValueError, AssertionError) as exc:
        print(f"Legacy summary unavailable: {exc}", flush=True)
    return report


def record_baseline_suite(config):
    from .baseline_variants import select_baseline, variants_for
    from .input_calibration import baseline_issue as calibrated_baseline_issue
    from .input_calibration import issue as calibrated_input_issue
    from .isolation import baseline_issue

    target = ROOT / "results" / "timing" / config.hardware / f"{config.baseline}.json"
    results = read_json(target, {})
    level = results.setdefault(f"level{config.level}", {})
    tasks, names, signatures, variants = [], {}, {}, {}
    for pid, path in select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track).items():
        base = task_payload(config, pid, task="baseline")
        signature = fingerprint(base)
        previous = level.get(path.name, {})
        if (config.resume and previous.get("suite_complete") and previous.get("metadata", {}).get("workflow_signature") == signature
                and not baseline_issue(config, previous, pid) and not calibrated_baseline_issue(config, previous, pid)):
            continue
        signatures[path.name] = signature
        variants[path.name] = {}
        for variant in variants_for(config, pid):
            payload = task_payload(config, pid, task="baseline")
            payload["eval_config"]["baseline_variant"] = variant
            payload["request_id"] = "kb-" + fingerprint(payload)
            key = f"baseline-{pid}-{variant}"
            names[key] = path.name, variant, pid
            tasks.append((key, payload))
    for key, result in evaluate_tasks(config, tasks):
        name, variant, pid = names[key]
        expected = next(payload for task_key, payload in tasks if task_key == key)
        if reason := calibrated_input_issue(result, expected):
            result = failure(reason)
        variants[name][variant] = result
        chosen = select_baseline(variants[name], config.cold_calls)
        chosen.setdefault("metadata", {})["workflow_signature"] = signatures[name]
        chosen["suite_complete"] = set(variants[name]) == set(variants_for(config, pid))
        chosen["requested_variants"] = list(variants_for(config, pid))
        level[name] = chosen
        atomic_json(target, results)
        print(f"[baseline] {name}/{variant}: {result.get('mean', result.get('metadata'))}", flush=True)
    atomic_json(target, results)
    return target


def paired_variant(config, pid):
    from .measurement import amortized

    reference = select_problems(config.level, problem_ids=[pid])[pid]
    data = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json", {})
    baseline = data.get(f"level{config.level}", {}).get(reference.name, {})
    options = baseline.get("variants", {"eager": baseline})
    valid = {name: value for name, value in options.items() if value.get("baseline_correctness") is True
             and type(value.get("mean")) in (int, float) and math.isfinite(value["mean"]) and value["mean"] > 0
             and type(value.get("cold_start_ms")) in (int, float) and math.isfinite(value["cold_start_ms"]) and value["cold_start_ms"] > 0}
    if not valid:
        raise ValueError("Paired timing requires a validated baseline portfolio")
    if config.score_mode == "headroom":
        return min(valid, key=lambda name: valid[name]["mean"])
    return min(valid, key=lambda name: amortized(valid[name]["mean"], valid[name]["cold_start_ms"], config.cold_calls))


def paired_payload(config, pid, sid, *, repeat=None, variant=None):
    variant = paired_variant(config, pid) if variant is None else variant
    if repeat is not None:
        from .final_verification import payload_for

        return payload_for(config, pid, sid, repeat, task="paired_timing", variant=variant)
    payload = task_payload(config, pid, sid, task="paired_timing")
    payload["eval_config"]["paired_baseline_variant"] = variant
    payload["request_id"] = "kb-" + fingerprint(payload)
    return payload


def run_paired(config):
    target = config.run_dir / "paired_results.json"
    results = read_json(target, {}) if config.resume else {}
    tasks = []
    selected = select_problems(config.level, config.start, config.end, suite=config.suite, suite_role=config.suite_role, track=config.task_track)
    for pid in selected:
        for sid in range(config.num_samples):
            for repeat in range(config.final_repeats or 1):
                key = f"{pid}_{sid}:{repeat}"
                try:
                    payload = paired_payload(config, pid, sid, repeat=repeat if config.final_repeats else None)
                except (ValueError, OSError, KeyError):
                    results[key] = failure("Paired timing source or baseline unavailable")
                    continue
                if config.resume and reusable(results.get(key, {}), payload):
                    continue
                tasks.append(("paired-" + key, payload))
    for key, result in evaluate_tasks(config, tasks):
        results[key.removeprefix("paired-")] = result
        atomic_json(target, results)
    atomic_json(target, results)
    return results

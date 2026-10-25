"""Source-bound empirical scaling certificates; finite-range evidence, not asymptotic proofs."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from pathlib import Path
from statistics import mean

from ..common import atomic_json, evaluator_digest, fingerprint, read_json
from .study_statistics import finite, interval

PROTOCOL = "kernelbench_pro_scaling_certificate_v1"
CONTRACT_PROTOCOL = "kernelbench_pro_scaling_contracts_v1"


def slope(sizes, times):
    x, y = [math.log(value) for value in sizes], [math.log(value) for value in times]
    center = mean(x)
    denominator = sum((value - center) ** 2 for value in x)
    if denominator <= 0:
        raise ValueError("Scaling regression needs distinct positive sizes")
    return sum((a - center) * (b - mean(y)) for a, b in zip(x, y)) / denominator


def fit(points, *, candidate_exponent, reference_exponent, tolerance, trials=2000, seed=42):
    if (not isinstance(points, list) or len(points) < 5 or type(trials) is not int or trials < 200
            or any(not finite(value) for value in (candidate_exponent, reference_exponent, tolerance))
            or not 0 <= candidate_exponent < reference_exponent or not 0 <= tolerance <= 0.5):
        raise ValueError("Scaling certificates require >=5 sizes, >=200 draws and reviewed separated exponents")
    sizes = [row["size"] for row in points]
    if (any(not finite(value) or value <= 0 for value in sizes) or len(set(sizes)) != len(sizes)
            or max(sizes) / min(sizes) < 16):
        raise ValueError("Scaling sizes must be unique and span at least 16x")
    for point in points:
        groups = point.get("groups", [])
        if not isinstance(groups, list) or len(groups) < 2 or any(len(group) < 10 for group in groups):
            raise ValueError("Each size needs >=2 independent groups and >=10 verified pairs per group")
        if any(not finite(row.get(key)) or row[key] <= 0 for group in groups for row in group
               for key in ("reference_ms", "candidate_ms", "control_ms")):
            raise ValueError("Scaling samples must contain finite positive verified times")

    def means(point, rng=None):
        groups = point["groups"] if rng is None else [rng.choice(point["groups"]) for _ in point["groups"]]
        chosen = [row for group in groups for row in (group if rng is None else [rng.choice(group) for _ in group])]
        return mean(row["reference_ms"] for row in chosen), mean(row["candidate_ms"] for row in chosen)

    observed = [means(point) for point in points]
    a_ref = slope(sizes, [value[0] for value in observed])
    a_cand = slope(sizes, [value[1] for value in observed])
    rng, draws_ref, draws_cand, gaps = random.Random(seed), [], [], []
    for _ in range(trials):
        sampled = [means(point, rng) for point in points]
        ref = slope(sizes, [value[0] for value in sampled])
        cand = slope(sizes, [value[1] for value in sampled])
        draws_ref.append(ref)
        draws_cand.append(cand)
        gaps.append(ref - cand)
    ref_ci, cand_ci, gap_ci = interval(draws_ref), interval(draws_cand), interval(gaps)
    matches = (candidate_exponent - tolerance <= cand_ci[0] <= cand_ci[1] <= candidate_exponent + tolerance
               and reference_exponent - tolerance <= ref_ci[0] <= ref_ci[1] <= reference_exponent + tolerance)
    return {"reference_exponent": a_ref, "candidate_exponent": a_cand, "reference_ci95": ref_ci, "candidate_ci95": cand_ci,
            "exponent_gap_ci95": gap_ci, "reviewed_candidate_exponent": candidate_exponent,
            "reviewed_reference_exponent": reference_exponent, "tolerance": tolerance,
            "matches_reviewed_regime": matches, "significantly_lower_than_reference": gap_ci[0] > 0,
            "passed": matches and gap_ci[0] > 0, "size_count": len(sizes), "size_span": max(sizes) / min(sizes),
            "trials": trials, "seed": seed, "inference": "fixed_size_log_mean_time_OLS; paired_repeat_then_pair_bootstrap",
            "scope": "finite_declared_regime; launch_floors_or_regime_changes_can_fail; no_universal_complexity_proof"}


def shape_regime(signatures, contract):
    axis = contract.get("axis")
    if (not isinstance(signatures, list) or not signatures or not isinstance(axis, dict)
            or any(type(axis.get(key)) is not int or axis[key] < 0 for key in ("input", "dimension"))
            or any(not isinstance(signature, list) or axis["input"] >= len(signature)
                   or not isinstance(signature[axis["input"]], dict)
                   or not isinstance(signature[axis["input"]].get("shape"), list)
                   or axis["dimension"] >= len(signature[axis["input"]]["shape"]) for signature in signatures)):
        raise ValueError("Scaling axis must name a tensor dimension in every input signature")
    rows = contract.get("shape_laws")
    if (not isinstance(rows, list) or any(not isinstance(row, dict)
            or any(type(row.get(key)) is not int or row[key] < 0 for key in ("input", "dimension"))
            or not finite(row.get("power")) for row in rows)):
        raise ValueError("Scaling laws require reviewed tensor dimensions and finite powers")
    laws = {(row["input"], row["dimension"]): row["power"] for row in rows}
    if len(laws) != len(contract["shape_laws"]) or laws.get((axis["input"], axis["dimension"])) != 1:
        raise ValueError("Scaling laws must uniquely identify the measured size axis with power 1")
    if any(type(i) is not int or type(j) is not int or i < 0 or j < 0 or not finite(power) or not 0 <= power <= 3
           for (i, j), power in laws.items()):
        raise ValueError("Invalid reviewed shape powers")
    sizes = [signature[axis["input"]]["shape"][axis["dimension"]] for signature in signatures]
    base = signatures[0]
    seen = set()
    for signature, size in zip(signatures, sizes):
        if len(signature) != len(base):
            raise ValueError("Scaling changed the input structure")
        for i, (first, current) in enumerate(zip(base, signature)):
            if "shape" not in first:
                if first != current:
                    raise ValueError("Unreviewed non-tensor input changed across sizes")
                continue
            if current.get("dtype") != first.get("dtype") or len(current.get("shape", [])) != len(first["shape"]):
                raise ValueError("Scaling changed dtype or tensor rank")
            for j, (a, b) in enumerate(zip(first["shape"], current["shape"])):
                seen.add((i, j))
                if type(a) is not int or type(b) is not int or min(a, b) <= 0 or not math.isclose(b / a, (size / sizes[0]) ** laws.get((i, j), 0), rel_tol=1e-9):
                    raise ValueError("Input shapes do not follow the reviewed scaling regime")
    if not set(laws) <= seen:
        raise ValueError("A scaling law names a nonexistent tensor dimension")
    return sizes


def read_contracts(path):
    document = read_json(Path(path))
    if not isinstance(document, dict) or document.get("protocol") != CONTRACT_PROTOCOL or not isinstance(document.get("tasks"), dict):
        raise ValueError("Missing reviewed scaling contracts")
    for key, contract in document["tasks"].items():
        review, instances = contract.get("review", {}), contract.get("instances", [])
        if (not review.get("reviewer") or not review.get("rationale") or not isinstance(instances, list) or len(instances) < 5
                or any(set(row) - {"dimensions", "memory_mb"} or not isinstance(row.get("dimensions"), dict)
                       or type(row.get("memory_mb")) is not int or row["memory_mb"] <= 0 for row in instances)
                or len({fingerprint(row) for row in instances}) != len(instances)):
            raise ValueError(f"Scaling contract needs a review and >=5 distinct explicit instances: {key}")
        # The fitting guard checks the numeric portion independently of empirical data.
        if (any(not finite(contract.get(name)) for name in ("candidate_exponent", "reference_exponent", "tolerance"))
                or not 0 <= contract["candidate_exponent"] < contract["reference_exponent"] or not 0 <= contract["tolerance"] <= 0.5):
            raise ValueError("Invalid reviewed scaling exponents")
    return document


def bind_contract(plan, task_key, contract, metadata):
    """Expected powers come from the reviewed D bound, never from a policy response."""
    from .headroom import certificate

    options = plan["verifier_config"]
    specs = read_json(Path(options["problem_specs"]))
    level, pid = map(int, task_key.split("/"))
    bound = specs.get(f"level{level}", {}).get(str(pid), {}).get("headroom_certificate", {})
    calibration = read_json(Path(options["device_calibration"]))
    info = certificate(bound, calibration, metadata, source_sha256=plan["tasks"][task_key]["reference_sha256"])
    declared = bound.get("design", {}).get("scaling", {})
    if (info["layer"] != "D" or contract.get("bound_certificate_sha256") != info["certificate_sha256"]
            or contract.get("reference_sha256") != plan["tasks"][task_key]["reference_sha256"]
            or any(contract.get(name) != declared.get(name) for name in
                   ("axis", "shape_laws", "candidate_exponent", "reference_exponent", "tolerance"))):
        raise ValueError("Scaling regime is not part of the current reviewed D/SOL algorithm certificate")
    return info


def point_evidence(config, directory, variant):
    from ..agents.rl_rewards import ROOT, select_reference, validate_reward
    from ..engine import paired_payload, task_payload
    from .paired_statistics import paired_summary
    from .paper_analysis import current_result, measurement_evidence

    directory = Path(directory)
    reward = validate_reward(directory / "reward.json")
    if not reward["interleaving"]["same_physical_device_verified"] or reward["components"][0]["qualified_status"] != "verified":
        raise ValueError("Scaling size lacks complete current final qualification")
    reference = select_reference(config)
    baseline = read_json(ROOT / "results/timing" / config.hardware / f"{config.baseline}.json")[f"level{config.level}"][reference.name]["variants"][variant]
    expected = task_payload(config, config.start, task="baseline")
    expected["eval_config"]["baseline_variant"] = variant
    if baseline.get("metadata", {}).get("workflow_signature") != fingerprint(expected) or measurement_evidence(baseline):
        raise ValueError("Scaling reference variant has stale measurement evidence")
    records = read_json(directory / "scaling_pairs.json")
    groups, signature, device = [], None, None
    for repeat in range(config.final_repeats):
        result = records[str(repeat)]
        payload = paired_payload(config, config.start, 0, repeat=repeat, variant=variant)
        metadata, pairs = result.get("metadata", {}), result.get("metadata", {}).get("paired_timing", {})
        if (result.get("compiled") is not True or result.get("correctness") is not True or current_result(result, payload)
                or pairs.get("baseline_variant") != variant or pairs.get("num_pairs") != config.paired_trials
                or any(pairs.get(name) is not True for name in ("output_consumed", "varied_inputs", "all_outputs_verified"))
                or metadata.get("input_signature") != baseline["input_signature"]
                or metadata.get("device_fingerprint") != baseline["device_fingerprint"]):
            raise ValueError("Fixed-reference scaling pairs lack matching source/device/workload evidence")
        recomputed = paired_summary(pairs["pairs"])
        if any(pairs.get(key) != recomputed[key] for key in ("protocol", "speedup_ci95", "reference_control_ci95", "reference_mean_ms", "candidate_mean_ms")):
            raise ValueError("Scaling timing summary differs from its raw paired samples")
        signature, device = metadata["input_signature"], metadata["device_fingerprint"]
        groups.append(pairs["pairs"])
    return {"groups": groups, "input_signature": signature, "device": device,
            "candidate_sha256": reward["source_sha256"][0], "reference_sha256": reward["reference_sha256"],
            "task_key": reward["task_key"],
            "reward_sha256": reward["receipt_sha256"]}


def collect(plan, task_key, source, contract, *, directory, anchor_metadata, dispatch=None, baseline_recorder=None):
    """Explicit execution path used only for enabled high-scoring terminal candidates."""
    from ..agents.rl_plan import task_config, verifier_inputs
    from ..agents.rl_rewards import _default_dispatch, evaluate_group
    from ..engine import paired_payload, record_baseline

    bind_contract(plan, task_key, contract, anchor_metadata)
    if plan["tasks"][task_key]["level"] != 5:
        raise ValueError("Automatic scaling sweeps require production41 explicit dimension instances")
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Scaling evidence cannot overwrite a previous attempt")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "contract.json", contract)
    atomic_json(root / "anchor_plan.json", plan)
    atomic_json(root / "anchor_metadata.json", anchor_metadata)
    points, files = [], {}
    variant = contract.get("baseline_variant", "eager")
    for index, instance in enumerate(contract["instances"]):
        dims = root / f"dimensions-{index}.json"
        atomic_json(dims, instance["dimensions"])
        case_plan = copy.deepcopy(plan)
        case_plan["output_directory"] = str(root / f"case-{index}")
        case_plan["training"].update(certificate_bonus=0.0, scaling_contracts="")
        case_plan["scaling_contracts_sha256"] = None
        # Every sweep owns an immutable baseline. Re-measuring a shared filename
        # would silently invalidate previously emitted training reward receipts.
        baseline_id = fingerprint({"plan": plan["plan_sha256"], "sweep": str(root), "instance": instance})[:20]
        case_plan["verifier_config"].update(task_dimensions=str(dims), task_memory_mb=instance["memory_mb"],
            score_mode="bounded_gain", problem_specs="", device_calibration="", baseline=f"scale_{baseline_id}")
        case_plan["verifier_inputs"] = verifier_inputs(case_plan["verifier_config"])
        case_plan["parent_training_plan_sha256"] = plan["plan_sha256"]
        case_plan["plan_sha256"] = fingerprint({key: value for key, value in case_plan.items() if key != "plan_sha256"})
        case_dir = root / f"case-{index}" / "evaluation"
        config = task_config(case_plan, task_key, directory=case_dir)
        (baseline_recorder or record_baseline)(config)
        evaluate_group(case_plan, task_key, [source], directory=case_dir, purpose="certificate", dispatch=dispatch)
        records = {}
        for repeat in range(config.final_repeats):
            payload = paired_payload(config, config.start, 0, repeat=repeat, variant=variant)
            result = (dispatch or _default_dispatch)(config, payload)
            result.setdefault("metadata", {}).update(workflow_signature=fingerprint(payload), evaluation_config=payload["eval_config"])
            records[str(repeat)] = result
            atomic_json(case_dir / "scaling_pairs.json", records)
        point = point_evidence(config, case_dir, variant)
        point["directory"] = str(case_dir.relative_to(root))
        points.append(point)
        for path in [dims, *(p for p in case_dir.rglob("*") if p.is_file() and p.suffix in {".json", ".py"})]:
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    sizes = shape_regime([point["input_signature"] for point in points], contract)
    for point, size in zip(points, sizes):
        point["size"] = size
    if len({fingerprint(point["device"]) for point in points}) != 1 or len({point["candidate_sha256"] for point in points}) != 1:
        raise ValueError("Scaling points changed physical device/software or candidate source")
    if any(point["task_key"] != task_key or point["reference_sha256"] != contract["reference_sha256"] for point in points):
        raise ValueError("Scaling points changed the reviewed reference task")
    result = fit(points, **{key: contract[key] for key in ("candidate_exponent", "reference_exponent", "tolerance")})
    for name in ("contract.json", "anchor_plan.json", "anchor_metadata.json"):
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    receipt = {"protocol": PROTOCOL, "evaluator_sha256": evaluator_digest(), "parent_plan_sha256": plan["plan_sha256"],
               "task_key": task_key, "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "contract_sha256": fingerprint(contract), "variant": variant, "points": points, "fit": result, "proof_files": files}
    receipt["receipt_sha256"] = fingerprint(receipt)
    atomic_json(root / "certificate.json", receipt)
    return receipt


def validate(path):
    from ..agents.rl_plan import validate_plan
    from ..config import RunConfig

    path = Path(path)
    receipt = read_json(path)
    if (not isinstance(receipt, dict) or receipt.get("protocol") != PROTOCOL or receipt.get("evaluator_sha256") != evaluator_digest()
            or receipt.get("receipt_sha256") != fingerprint({key: value for key, value in receipt.items() if key != "receipt_sha256"})):
        raise ValueError("Missing or stale scaling certificate")
    root = path.parent
    for name, digest in receipt["proof_files"].items():
        target = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or target.is_symlink() or not target.is_file()
                or hashlib.sha256(target.read_bytes()).hexdigest() != digest):
            raise ValueError("Scaling evidence changed")
    contract = read_json(root / "contract.json")
    plan = read_json(root / "anchor_plan.json")
    validate_plan(plan)
    if plan["plan_sha256"] != receipt["parent_plan_sha256"]:
        raise ValueError("Scaling certificate belongs to a different training plan")
    bind_contract(plan, receipt["task_key"], contract, read_json(root / "anchor_metadata.json"))
    points = []
    for stored in receipt["points"]:
        directory = root / stored["directory"]
        config = RunConfig(**read_json(directory / "run_config.json")).validate()
        point = point_evidence(config, directory, receipt["variant"])
        if any(point[key] != stored[key] for key in point):
            raise ValueError("Scaling point no longer matches its native evidence")
        points.append(point)
    sizes = shape_regime([point["input_signature"] for point in points], contract)
    for point, size in zip(points, sizes):
        point["size"] = size
    if (fingerprint(contract) != receipt["contract_sha256"] or any(point["candidate_sha256"] != receipt["candidate_sha256"] for point in points)
            or any(point["task_key"] != receipt["task_key"] or point["reference_sha256"] != contract["reference_sha256"] for point in points)
            or len({fingerprint(point["device"]) for point in points}) != 1
            or fit(points, **{key: contract[key] for key in ("candidate_exponent", "reference_exponent", "tolerance")}) != receipt["fit"]):
        raise ValueError("Scaling fit or identity differs from the verified size series")
    return receipt

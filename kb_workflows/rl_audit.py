"""Offline RL reward reliability, attack and held-out analysis; no experiment launcher."""

from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import mean

from .common import fingerprint, kernel_name
from .rl_rewards import validate_reward
from .study_statistics import finite, interval


def intraclass_correlation(matrix):
    """ICC(A,1)/ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    if (not isinstance(matrix, list) or len(matrix) < 2 or not isinstance(matrix[0], list) or len(matrix[0]) < 2
            or any(len(row) != len(matrix[0]) or any(not finite(value) for value in row) for row in matrix)):
        raise ValueError("ICC requires a finite rectangular matrix with >=2 kernels and >=2 repetitions")
    n, k = len(matrix), len(matrix[0])
    grand = mean(value for row in matrix for value in row)
    row_means = list(map(mean, matrix))
    column_means = [mean(row[column] for row in matrix) for column in range(k)]
    between = k * sum((value - grand) ** 2 for value in row_means) / (n - 1)
    repeats = n * sum((value - grand) ** 2 for value in column_means) / (k - 1)
    error = sum((value - row_means[i] - column_means[j] + grand) ** 2
                for i, row in enumerate(matrix) for j, value in enumerate(row)) / ((n - 1) * (k - 1))
    denominator = between + (k - 1) * error + k * (repeats - error) / n
    return (between - error) / denominator if denominator > 0 else None


def wilson(successes, total):
    if type(total) is not int or type(successes) is not int or not 0 <= successes <= total or total == 0:
        raise ValueError("Binomial interval requires a nonempty valid count")
    z = 1.959963984540054
    rate, divisor = successes / total, 1 + z * z / total
    center = (rate + z * z / (2 * total)) / divisor
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / divisor
    return [max(0, center - half), min(1, center + half)]


def reward_sample(item):
    path, sid = Path(item["receipt"]), item["sample_id"]
    receipt = validate_reward(path)
    if type(sid) is not int or not 0 <= sid < receipt["candidate_count"]:
        raise ValueError("Reward sample outside the declared group")
    return receipt, receipt["components"][sid], receipt["source_sha256"][sid]


def reliability(specification):
    """Anchors must be independent measurements; they are never inferred from the tested ordering."""
    kernels = specification.get("kernels", [])
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("Reliability audit needs declared kernel identities and repeated reward receipts")
    matrix, anchors, identities, rows, receipt_sets = [], [], set(), [], []
    for kernel in kernels:
        values, receipts, identity = [], set(), None
        for sample in kernel["repetitions"]:
            receipt, component, source = reward_sample(sample)
            current = (receipt["task_key"], source)
            if source is None or identity is not None and current != identity:
                raise ValueError("A reliability kernel must retain its task and exact candidate source")
            identity = current
            if receipt["receipt_sha256"] in receipts:
                raise ValueError("Copied/reused reward groups do not constitute independent remeasurements")
            receipts.add(receipt["receipt_sha256"])
            values.append(component["qualified"])
        if not values or identity in identities:
            raise ValueError("Missing repetitions or duplicated kernel identity")
        identities.add(identity)
        matrix.append(values)
        receipt_sets.append(receipts)
        anchor = kernel.get("independent_runtime_ms")
        if anchor is not None and (not finite(anchor) or anchor <= 0 or not kernel.get("anchor_review")):
            raise ValueError("Ordering anchors require positive independent runtimes and a review/source description")
        anchors.append(anchor)
        rows.append({"kernel": kernel["kernel"], "task_key": identity[0], "source_sha256": identity[1],
                     "repetitions": len(values), "mean_reward": mean(values), "anchor_review": kernel.get("anchor_review")})
    same_count = len({len(row) for row in matrix}) == 1
    if not same_count:
        raise ValueError("All reliability kernels need the same number of declared repetitions")
    icc = intraclass_correlation(matrix) if len(matrix) >= 2 and same_count and len(matrix[0]) >= 2 else None
    events, pairs = [], []
    for a in range(len(matrix)):
        for b in range(a + 1, len(matrix)):
            if anchors[a] is None or anchors[b] is None or max(anchors[a], anchors[b]) / min(anchors[a], anchors[b]) < 1.05:
                continue
            if rows[a]["task_key"] != rows[b]["task_key"]:
                continue  # Different problem rewards and bounds are not a kernel-speed ordering.
            expected = 1 if anchors[a] < anchors[b] else -1
            outcomes = [float((x - y) * expected > 0) for x, y in zip(matrix[a], matrix[b], strict=True)]
            events.extend(outcomes)
            pairs.append({"a": rows[a]["kernel"], "b": rows[b]["kernel"], "events": len(outcomes), "correct": sum(outcomes),
                          "ties_count_as_correct": False})
    enough = same_count and min(map(len, matrix)) >= 30
    return {"kernels": rows, "icc_absolute_single": icc, "icc_target": 0.95,
            "icc_target_met": enough and icc is not None and icc >= 0.95,
            "minimum_30_repetitions_met": enough, "single_kernel_icc_identifiable": False,
            "ordering": {"accuracy": mean(events) if events else None, "target": 0.98,
                         "point_target_met": enough and bool(events) and mean(events) >= 0.98,
                         "wilson_ci95": wilson(int(sum(events)), len(events)) if events else None, "pairs": pairs,
                         "interval_assumption": "independent_Bernoulli_events; shared_kernels_or_groups_violate_this_assumption",
                         "statistical_target_certified": False},
            "receipt_groups": sorted(set().union(*receipt_sets)), "experiments_started": False}


def attacks(items):
    if not isinstance(items, list) or not items or len({row["attack"] for row in items}) != len(items):
        raise ValueError("Attack audit requires distinct predeclared attack cases")
    rows = []
    for item in items:
        if item.get("expected_invalid") is not True or not item.get("rationale"):
            raise ValueError("Attack expectations must explain why a candidate is invalid")
        receipt, component, source = reward_sample(item)
        expected = item.get("expected_block_checks")
        if not isinstance(expected, list) or not expected or any(not isinstance(name, str) for name in expected):
            raise ValueError("Attacks require the predeclared checks that should reject them")
        if not component["qualification_accepted"] and not set(expected) & set(component.get("failed_checks", [])):
            raise ValueError("Attack lacks evidence of its expected check; infrastructure failures cannot count as hack resistance")
        rows.append({"attack": item["attack"], "task_key": receipt["task_key"], "source_sha256": source,
                     "qualification_accepted": component["qualification_accepted"], "positive_reward": component["qualified"] > 0,
                     "raw_positive_reward": component["raw_speedup"] > 0, "rationale": item["rationale"],
                     "receipt_sha256": receipt["receipt_sha256"]})
    accepted = sum(row["qualification_accepted"] for row in rows)
    positive = sum(row["positive_reward"] for row in rows)
    ci = wilson(accepted, len(rows))
    return {"attacks": rows, "declared_cases": len(rows), "qualification_success_rate": accepted / len(rows),
            "positive_reward_rate": positive / len(rows), "qualification_rate_ci95": ci, "target_max": 0.02,
            "point_target_met": accepted / len(rows) <= 0.02, "interval_target_met": ci[1] <= 0.02,
            "assumption": "independent_predeclared_attack_cases; no_claim_of_exhaustive_hack_resistance",
            "experiments_started": False}


def public_hidden_gap(public_directory, final_directory, *, trials=2000, seed=42):
    from .study_data import collect_run

    if type(trials) is not int or trials < 100:
        raise ValueError("Gap confidence intervals need >=100 bootstrap draws")
    public, final = [collect_run(path, session="reward-gap") for path in (public_directory, final_directory)]
    contracts = [dict(row["evaluation_contract"]) for row in (public, final)]
    if contracts[0].pop("hidden_tests") is not False or contracts[1].pop("hidden_tests") is not True or contracts[0] != contracts[1]:
        raise ValueError("Public/final audits must share the evaluation contract and differ only in hidden checks")
    if public["reference_sha256"] != final["reference_sha256"] or public["layers"] != final["layers"]:
        raise ValueError("Public/final task populations or strata differ")
    diffs = []
    for task in public["tasks"]:
        level, pid = map(int, task.split("/"))
        a, b = public["tasks"][task], final["tasks"][task]
        if len(a) != len(b):
            raise ValueError("Public/final samples differ")
        for sid in range(len(a)):
            name = kernel_name(level, pid, sid)
            paths = [Path(directory) / name for directory in (public_directory, final_directory)]
            if any(not path.is_file() for path in paths) or paths[0].read_bytes() != paths[1].read_bytes():
                raise ValueError("Public/final comparison must regrade the identical candidate source")
        diffs.append([x - y for x, y in zip(a, b)])
    rng = random.Random(seed)
    draws = [mean(mean(rng.choice(row) for _ in row) for row in (rng.choice(diffs) for _ in diffs)) for _ in range(trials)]
    ci = interval(draws)
    return {"mean_public_minus_final": mean(map(mean, diffs)), "ci95": ci, "target_max": 0.05,
            "interval_target_met": ci[1] <= 0.05, "task_count": len(diffs), "failures_retained": True,
            "ci_method": "paired_task_then_slot_percentile_bootstrap", "source_receipts": [row["receipt_sha256"] for row in (public, final)],
            "experiments_started": False}


def analyze(document):
    if set(document) - {"reliability", "attacks", "public_hidden"}:
        raise ValueError("Unknown reward audit sections")
    report = {"kind": "kernelbench_pro_rl_reward_audit_v1", "experiments_started": False}
    for key in ("reliability", "attacks", "public_hidden"):
        if key not in document:
            report[key] = {"status": "not_measured"}
        elif key == "reliability":
            report[key] = reliability(document[key])
        elif key == "attacks":
            report[key] = attacks(document[key])
        else:
            report[key] = public_hidden_gap(document[key]["public"], document[key]["final"])
    report["receipt_sha256"] = fingerprint(report)
    return report

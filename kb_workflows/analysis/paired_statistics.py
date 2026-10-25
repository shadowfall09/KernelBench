"""Paired log-speedup intervals with an explicit reference-vs-reference control."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, median


def paired_summary(pairs, *, bootstrap_trials=2000, seed=42):
    if len(pairs) < 10 or type(bootstrap_trials) is not int or bootstrap_trials < 100:
        raise ValueError("Paired inference needs at least 10 pairs and 100 bootstrap draws")
    for pair in pairs:
        if any(type(pair.get(key)) not in (int, float) or not math.isfinite(pair[key]) or pair[key] <= 0
               for key in ("reference_ms", "candidate_ms", "control_ms")):
            raise ValueError("Paired samples must contain finite positive times")
    logs = [math.log(row["reference_ms"] / row["candidate_ms"]) for row in pairs]
    controls = [math.log(row["reference_ms"] / row["control_ms"]) for row in pairs]
    rng = random.Random(seed)
    groups = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups[pair.get("repeat", 0)].append(index)
    labels = list(groups)
    gains, noise = [], []
    for _ in range(bootstrap_trials):
        indices = []
        for _ in labels:
            group = groups[rng.choice(labels)]
            indices.extend(rng.choice(group) for _ in group)
        gains.append(math.exp(mean(logs[index] for index in indices)))
        noise.append(math.exp(mean(controls[index] for index in indices)))
    gains.sort()
    noise.sort()
    low, high = int(0.025 * (bootstrap_trials - 1)), int(0.975 * (bootstrap_trials - 1))
    gain_ci, noise_ci = [gains[low], gains[high]], [noise[low], noise[high]]
    return {"protocol": "balanced_paired_wall_v1", "pairs": pairs, "num_pairs": len(pairs),
            "speedup_geomean": math.exp(mean(logs)), "speedup_median": median(math.exp(value) for value in logs),
            "speedup_ci95": gain_ci, "reference_control_ci95": noise_ci,
            "reference_mean_ms": mean(row["reference_ms"] for row in pairs),
            "candidate_mean_ms": mean(row["candidate_ms"] for row in pairs),
            "control_mean_ms": mean(row["control_ms"] for row in pairs),
            "significantly_faster": gain_ci[0] > max(1.0, noise_ci[1]),
            "ci_method": "repeat_then_pair_percentile_bootstrap_of_mean_log_ratio; exchangeable_groups_and_pairs",
            "independent_repeat_groups": len(groups),
            "bootstrap_trials": bootstrap_trials, "bootstrap_seed": seed,
            "shared_device_drift_eliminated": False}

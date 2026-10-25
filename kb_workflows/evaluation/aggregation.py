"""Compare aggregation rules on the same declared slots without changing scores."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median


def sensitivity(report):
    ids, count = report["problem_ids"], report["declared_samples"]
    if not ids or len(ids) != len(set(ids)) or type(count) is not int or count < 1:
        raise ValueError("Invalid declared sample population")
    slots = {pid: [0.0] * count for pid in ids}
    layers, seen = {}, set()
    for row in report["details"]:
        pid, sid = row["problem_id"], row["sample_id"]
        if pid not in slots or type(sid) is not int or not 0 <= sid < count or (pid, sid) in seen:
            raise ValueError("Duplicate or out-of-population score slot")
        seen.add((pid, sid))
        layer = row.get("layer", "unclassified")
        if pid in layers and layers[pid] != layer:
            raise ValueError("One problem cannot belong to different layers across samples")
        layers[pid] = layer
        value = row.get("score", 0)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Scores must be finite and bounded")
        slots[pid][sid] = value if row.get("status") == "verified" else 0.0
    policies = {}
    for name, reducer in (("mean", mean), ("median", median), ("best_of_n", max)):
        grouped = defaultdict(list)
        for pid, values in slots.items():
            grouped[layers.get(pid, "unclassified")].append(reducer(values))
        by_layer = {layer: mean(values) for layer, values in grouped.items()}
        policies[name] = {"by_layer": by_layer, "macro": mean(by_layer.values())}
    return {"purpose": "aggregation_ablation_on_identical_saved_slots", "official_policy": "mean",
            "task_count": len(ids), "declared_slots": len(ids) * count, "missing_slots": len(ids) * count - len(seen),
            "failures_and_missing_are_zero": True, "policies": policies,
            "best_minus_mean": policies["best_of_n"]["macro"] - policies["mean"]["macro"],
            "generation_budget": report.get("generation_budget", {}), "requires_additional_model_calls": False,
            "does_not_compare_generation_methods_or_unequal_sample_budgets": True}

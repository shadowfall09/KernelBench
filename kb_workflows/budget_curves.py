"""Score/resource curves from independent saved runs, never hidden best-so-far traces."""

from __future__ import annotations

import random
from statistics import mean

from .common import fingerprint
from .study_data import assert_session_identity, study_digest, validate_record
from .study_statistics import aggregate, finite, interval, paired_draw

AXES = {"declared_wall_seconds": None, "declared_candidates": None,
        "observed_total_tokens": "total_tokens", "observed_output_tokens": "output_tokens",
        "observed_model_calls": "model_calls", "observed_estimated_cost_usd": "estimated_cost_usd",
        "observed_generation_wall_seconds": "generation_wall_seconds"}
VARY = {"wall": "wall_limit_seconds_per_sample", "candidates": "candidate_limit_per_sample"}


def area(points, left, right):
    """Trapezoidal area of the actual curve, restricted to an interpolation domain."""
    points = sorted(points)
    if len(points) < 2 or not points[0][0] <= left < right <= points[-1][0]:
        raise ValueError("Area requires a shared positive-width domain without extrapolation")
    if len({x for x, _ in points}) != len(points):
        raise ValueError("Two distinct protocol points have the same observed resource value")
    def interpolate(x):
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        raise ValueError("Interpolation point is outside observed domain")
    grid = [left, *(x for x, _ in points if left < x < right), right]
    return sum((b - a) * (interpolate(a) + interpolate(b)) / 2 for a, b in zip(grid, grid[1:]))


def point_summary(point, layers, axis, aggregation, trials, seed):
    sessions = point["sessions"]
    population = sorted(layers)
    if not sessions or any(set(values) != set(population) for values in sessions.values()):
        raise ValueError("Every budget/session must declare the identical task population")
    subject = {"sessions": sessions}
    means = {key: mean(mean(session[key]) for session in sessions.values()) for key in population}
    score = aggregate(list(means.items()), layers, aggregation)
    rng = random.Random(seed)
    draws = [paired_draw({"point": subject}, population, set(population), layers, aggregation, rng)[0][0]
             for _ in range(trials)]
    resource_field = AXES[axis]
    if resource_field is None:
        field = VARY["wall" if axis == "declared_wall_seconds" else "candidates"]
        x = point["generation_protocol"].get(field)
    else:
        values = [record["resource_totals"].get(resource_field) for record in point["records"]]
        slots = sum(record["declared_slots"] for record in point["records"])
        x = sum(values) / slots if all(finite(value) and value >= 0 for value in values) else None
    if x is not None and (not finite(x) or x < 0):
        raise ValueError("Budget coordinates must be nonnegative finite values")
    return {"x": x, "score": score, "score_ci95": interval(draws), "session_count": len(sessions),
            "declared_tasks": len(population), "declared_slots": sum(len(slots) for values in sessions.values() for slots in values.values()),
            "generation_protocol": point["generation_protocol"], "source_receipts": [record["receipt_sha256"] for record in point["records"]],
            "budget_status": "observed_or_declared" if x is not None else "unknown; excluded_from_curve_not_imputed_zero",
            "formal_comparison_eligible": all(record["formal_comparison_eligible"] for record in point["records"])}


def budget_curves(records, *, axis="declared_wall_seconds", vary="wall", aggregation="task_mean", trials=1000,
                  seed=42, audit_only=False):
    if (axis not in AXES or vary not in VARY or aggregation not in {"task_mean", "layer_macro"}
            or type(trials) is not int or trials < 100):
        raise ValueError("Unknown axis/vary/aggregation or insufficient bootstrap trials")
    if axis.startswith("declared_") and ((axis == "declared_wall_seconds") != (vary == "wall")):
        raise ValueError("The declared x axis must be the varied protocol limit")
    groups, evidence_sessions = {}, {}
    for record in records:
        validate_record(record)
        assert_session_identity(record, evidence_sessions)
        if not record["formal_comparison_eligible"] and not audit_only:
            raise ValueError("Incomplete attribution: use audit-only for exploratory curves")
        contract = {"evaluation": record["evaluation_contract"], "metric": record["metric"],
                    "speedup_threshold": record["speedup_threshold"]}
        group = groups.setdefault(fingerprint(contract), {"contract": contract, "series": {}, "layers": {}, "reference_sha256": {}})
        method = dict(record["generation_protocol"])
        limit = method.pop(VARY[vary], None)
        if not finite(limit) or limit <= 0:
            raise ValueError("This method cannot certify the selected varying limit")
        descriptor = {"model_identity": record["model_identity"], "method": method}
        series = group["series"].setdefault(fingerprint(descriptor), {"descriptor": descriptor, "points": {}})
        point = series["points"].setdefault(limit, {"generation_protocol": record["generation_protocol"], "sessions": {}, "records": []})
        session = point["sessions"].setdefault(record["session"], {})
        if set(session) & set(record["tasks"]):
            raise ValueError("Duplicate budget/subject/session task; independent runs cannot be cherry-picked")
        session.update(record["tasks"])
        point["records"].append(record)
        for key in record["tasks"]:
            if key in group["layers"] and (group["layers"][key] != record["layers"][key] or
                                            group["reference_sha256"][key] != record["reference_sha256"][key]):
                raise ValueError("Curve points use incompatible task sources or strata")
            group["layers"][key], group["reference_sha256"][key] = record["layers"][key], record["reference_sha256"][key]
    output = []
    for group in groups.values():
        series_rows, domains = [], []
        for series in group["series"].values():
            points = [point_summary(point, group["layers"], axis, aggregation, trials, seed)
                      for _, point in sorted(series["points"].items())]
            known = [(point["x"], point["score"]) for point in points if point["x"] is not None]
            unique = len({x for x, _ in known}) == len(known)
            domains.append((min(x for x, _ in known), max(x for x, _ in known)) if len(known) >= 2 and unique else None)
            series_rows.append({"descriptor": series["descriptor"], "points": points,
                                "auc_status": "pending_shared_domain" if domains[-1] else "insufficient_or_duplicate_observed_coordinates"})
        common = (max(domain[0] for domain in domains), min(domain[1] for domain in domains)) if domains and all(domains) else None
        if common and common[0] < common[1]:
            for series in series_rows:
                points = [(point["x"], point["score"]) for point in series["points"] if point["x"] is not None]
                series.update(auc_normalized=area(points, *common) / (common[1] - common[0]), auc_status="shared_interpolation_domain")
        else:
            common = None
            for series in series_rows:
                series["auc_normalized"] = None
                if series["auc_status"] == "pending_shared_domain":
                    series["auc_status"] = "no_shared_interpolation_domain"
        output.append({"contract": group["contract"], "series": series_rows, "shared_auc_domain": common,
                       "problem_ids": sorted(group["layers"])})
    return {"protocol": "score_budget_curves_v1", "axis": axis, "varied_limit": vary, "aggregation": aggregation,
            "analysis_implementation_sha256": study_digest(),
            "axis_normalization": "per_declared_sample_slot" if AXES[axis] else "per_sample_declared_cap",
            "groups": output, "unknown_costs_imputed_zero": False, "monotonic_envelope_or_best_of_history_used": False,
            "gpu_billing_certified": False, "equal_device_time_certified": False,
            "audit_only": audit_only, "x_coordinate_uncertainty_estimated": False,
            "bootstrap": {"trials": trials, "seed": seed, "method": "tasks_then_sessions_then_declared_slots"},
            "experiments_started_by_analysis": False}


def plot_curves(report, output):
    """Optional static export. Raw point order and nonmonotonic results are preserved."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report["groups"]:
        raise ValueError("No curves to plot")
    figure, axes = plt.subplots(len(report["groups"]), 1, squeeze=False, figsize=(8, 4 * len(report["groups"])))
    for axis, group in zip(axes[:, 0], report["groups"]):
        for series in group["series"]:
            points = sorted((point for point in series["points"] if point["x"] is not None), key=lambda point: point["x"])
            if not points:
                continue
            desc = series["descriptor"]
            label = f"{desc['model_identity']} / {desc['method'].get('scaffold')} / {desc['method'].get('mode')}"
            line, = axis.plot([point["x"] for point in points], [point["score"] for point in points], marker="o", label=label)
            # Draw exact interval endpoints even when a percentile interval excludes its point estimate.
            axis.vlines([point["x"] for point in points], [point["score_ci95"][0] for point in points],
                        [point["score_ci95"][1] for point in points], color=line.get_color())
        axis.set(xlabel=report["axis"] + " (" + report["axis_normalization"] + ")", ylabel=group["contract"]["metric"], ylim=(-0.02, 1.02))
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)

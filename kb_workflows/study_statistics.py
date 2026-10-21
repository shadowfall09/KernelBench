"""Fixed-population selection analysis with ties and hierarchical resampling."""

from __future__ import annotations

import math
import random
from itertools import combinations
from statistics import NormalDist, mean, pstdev


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def ranks(values):
    """Average ranks for exact ties; larger scores receive larger ranks."""
    if not values or not all(map(finite, values)):
        raise ValueError("Ranks need finite observations")
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result, index = [0.0] * len(values), 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        for position in ordered[index:end]:
            result[position] = (index + 1 + end) / 2
        index = end
    return result


def pearson(first, second):
    if len(first) != len(second) or len(first) < 2 or not all(map(finite, [*first, *second])):
        raise ValueError("Correlation requires equal finite vectors with at least two entries")
    x, y = mean(first), mean(second)
    dx, dy = [value - x for value in first], [value - y for value in second]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(dx, dy)) / denominator)) if denominator else None


def spearman(first, second):
    return pearson(ranks(first), ranks(second))


def kendall_b(first, second):
    if len(first) != len(second) or len(first) < 2 or not all(map(finite, [*first, *second])):
        raise ValueError("Kendall correlation requires equal finite vectors")
    concordant = discordant = tied_x = tied_y = 0
    for i, j in combinations(range(len(first)), 2):
        x, y = first[i] - first[j], second[i] - second[j]
        if x == 0 and y == 0:
            continue
        if x == 0:
            tied_x += 1
        elif y == 0:
            tied_y += 1
        elif x * y > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + tied_x) * (concordant + discordant + tied_y))
    return (concordant - discordant) / denominator if denominator else None


def interval(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    def quantile(fraction):
        position = fraction * (len(values) - 1)
        left = math.floor(position)
        return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)
    return [quantile(0.025), quantile(0.975)]


def paired_binary_design(tasks, discordance, *, alpha=0.05, power=0.8):
    """Normal design approximation for paired {-1,0,+1} task differences."""
    if (type(tasks) is not int or tasks < 2 or not finite(discordance) or not 0 <= discordance <= 1
            or not finite(alpha) or not 0 < alpha < 1 or not finite(power) or not 0.5 < power < 1):
        raise ValueError("Invalid paired binary design parameters")
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    gap = math.sqrt(z * z * discordance / (tasks + z * z))
    feasible = discordance > 0 and gap <= discordance
    return {"tasks": tasks, "discordance": discordance, "alpha": alpha, "target_power": power,
            "unit": "one_paired_binary_outcome_per_independent_task",
            "minimum_gap_normal_approximation": gap if feasible else None,
            "reason": "normal_approximation_at_fixed_discordance" if feasible else "no_feasible_gap_at_fixed_discordance",
            "independent_tasks_assumed": True, "operator_family_correlation_adjusted": False,
            "empirical_power_verified": False}


def validate_subjects(subjects, layers):
    if not isinstance(subjects, dict) or len(subjects) < 2 or not layers:
        raise ValueError("A discrimination study needs at least two subjects and a declared task population")
    population, slot_counts = set(layers), {}
    for subject in subjects.values():
        sessions = subject.get("sessions", {})
        if not sessions:
            raise ValueError("Every subject needs explicit sessions")
        for session in sessions.values():
            if set(session) != population:
                raise ValueError("Every session must declare the identical task population; missing runs are not missing slots")
            for key, slots in session.items():
                if (not isinstance(slots, list) or not slots or
                        any(not finite(value) or not 0 <= value <= 1 for value in slots)):
                    raise ValueError("Each task needs declared finite score slots, including failures as zeros")
                if key in slot_counts and slot_counts[key] != len(slots):
                    raise ValueError("Ranking comparisons require the same declared sample budget")
                slot_counts[key] = len(slots)


def task_means(subject):
    sessions = list(subject["sessions"].values())
    return {key: mean(mean(session[key]) for session in sessions) for key in sessions[0]}


def aggregate(rows, layers, policy):
    if policy == "task_mean":
        return mean(value for _, value in rows)
    if policy != "layer_macro":
        raise ValueError("Unknown aggregation policy")
    grouped = {}
    for key, value in rows:
        grouped.setdefault(layers[key], []).append(value)
    return mean(mean(values) for values in grouped.values())


def paired_draw(subjects, population, selected, layers, policy, rng):
    """Keep membership/layer counts fixed and reuse selected draws in the full score."""
    strata = {}
    for key in population:
        strata.setdefault((layers[key] if policy == "layer_macro" else "all", key in selected), []).append(key)
    draws = [rng.choice(keys) for keys in strata.values() for _ in keys]
    full, subset = [], []
    for subject in subjects.values():
        sessions = list(subject["sessions"].values())
        sampled_sessions = [rng.choice(sessions) for _ in sessions]
        rows = []
        for key in draws:
            value = mean(mean(rng.choice(session[key]) for _ in session[key]) for session in sampled_sessions)
            rows.append((key, value))
        full.append(aggregate(rows, layers, policy))
        subset.append(aggregate([(key, value) for key, value in rows if key in selected], layers, policy))
    return full, subset


def selection_analysis(subjects, layers, selected, *, aggregation="task_mean", trials=1000, seed=42,
                       redundancy_threshold=0.98):
    validate_subjects(subjects, layers)
    if (type(trials) is not int or trials < 100 or not 0 < redundancy_threshold <= 1
            or aggregation not in {"task_mean", "layer_macro"}):
        raise ValueError("Use >=100 bootstrap trials, a valid aggregation, and a correlation threshold in (0,1]")
    selected = list(selected)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(layers):
        raise ValueError("Selection must be a nonempty unique subset of the declared population")
    population, names = sorted(layers), list(subjects)
    selected_set = set(selected)
    means = {name: task_means(subjects[name]) for name in names}
    full = [aggregate(list(means[name].items()), layers, aggregation) for name in names]
    subset = [aggregate([(key, means[name][key]) for key in selected], layers, aggregation) for name in names]
    rng, full_draws, subset_draws = random.Random(seed), [], []
    for _ in range(trials):
        a, b = paired_draw(subjects, population, selected_set, layers, aggregation, rng)
        full_draws.append(a)
        subset_draws.append(b)
    agreements = {"spearman": spearman(full, subset), "kendall_tau_b": kendall_b(full, subset)}
    for key, function in (("spearman", spearman), ("kendall_tau_b", kendall_b)):
        values = [function(a, b) for a, b in zip(full_draws, subset_draws)]
        agreements[key + "_ci95"] = interval(values)
        agreements[key + "_defined_draws"] = sum(value is not None for value in values)
    pairwise = []
    for i, j in combinations(range(len(names)), 2):
        full_gap, subset_gap = full[i] - full[j], subset[i] - subset[j]
        pairwise.append({"subjects": [names[i], names[j]], "full_gap": full_gap, "selected_gap": subset_gap,
                         "full_gap_ci95": interval([row[i] - row[j] for row in full_draws]),
                         "selected_gap_ci95": interval([row[i] - row[j] for row in subset_draws]),
                         "observed_strict_order_reversed": full_gap * subset_gap < 0})
    redundancy = []
    if len(names) >= 3:
        for first, second in combinations(population, 2):
            correlation = pearson([means[name][first] for name in names], [means[name][second] for name in names])
            if correlation is not None and correlation >= redundancy_threshold:
                redundancy.append({"tasks": [first, second], "across_subject_correlation": correlation})
    return {"protocol": "selection_discrimination_v1", "aggregation": aggregation,
            "population": population, "selected": sorted(selected), "subject_order": names,
            "scores": {name: {"full": full[i], "selected": subset[i],
                              "full_ci95": interval([row[i] for row in full_draws]),
                              "selected_ci95": interval([row[i] for row in subset_draws]),
                              "session_count": len(subjects[name]["sessions"])} for i, name in enumerate(names)},
            "agreement": agreements, "pairwise": pairwise,
            "discrimination": {"full_subject_std": pstdev(full), "selected_subject_std": pstdev(subset),
                               "full_range": max(full) - min(full), "selected_range": max(subset) - min(subset)},
            "redundancy": {"pairs": redundancy, "minimum_subjects": 3, "threshold": redundancy_threshold,
                           "purpose": "exploratory_only; not_automatic_task_removal"},
            "bootstrap": {"trials": trials, "seed": seed,
                          "method": "membership_and_layer_stratified_tasks; independent_subject_sessions; within_task_slots",
                          "selection_algorithm_uncertainty_measured": False,
                          "operator_family_correlation_adjusted": False,
                          "single_session_cannot_measure_between_session_variation": True,
                          "shared_device_drift_eliminated": False},
            "historical_seven_model_ranking_reproduced": False, "asymptotic_small_model_count_pvalues_used": False}

"""Reviewed framework-only equivalent rewrites, tied to reference source hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("baseline_rewrites.json")
PROTOCOL = "kernelbench_pro_baseline_recipe_v1"


def entry_for(level, pid, source):
    item = json.loads(MANIFEST.read_text()).get(f"{level}/{pid}")
    if item and item["reference_sha256"] != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Baseline rewrite reference changed; review semantic equivalence")
    return item


def rewrite_for(level, pid, source):
    item = entry_for(level, pid, source)
    return item.get("rewrite") if item else None


def extension_options(level, pid, source):
    item = entry_for(level, pid, source)
    if item is None:
        return {}
    options = {}
    for index, name in enumerate(item.get("rewrites", [item["rewrite"]] if item.get("rewrite") else [])):
        variant = "library_rewrite" if index == 0 else "library_rewrite_" + name
        options[variant] = {"rewrite": name, "backend_policy": {},
                            "value_domain": item.get("rewrite_domains", {}).get(name, "declared_nominal_task_roles")}
    for variant, policy in item.get("backend_policies", {}).items():
        options[variant] = {"rewrite": None, "backend_policy": policy}
    return {variant + suffix: {"protocol": PROTOCOL, "reference_sha256": item["reference_sha256"],
                "task": [level, pid],
                "variant": variant + suffix, "outer_graph": bool(suffix), **spec}
            for variant, spec in options.items() for suffix in ("", "_cuda_graph")}


def plan_for(level, pid, source, variant):
    from .baseline_variants import EXTENDED_VARIANTS

    options = extension_options(level, pid, source)
    if variant in EXTENDED_VARIANTS:
        return None
    if variant not in options:
        raise ValueError("No source-bound baseline recipe for requested variant")
    return options[variant]


def issue(result, payload):
    if result.get("baseline_correctness") is not True or payload["eval_config"].get("device_backend") == "neuron":
        return None
    variant = payload["eval_config"].get("baseline_variant", "eager")
    try:
        expected = plan_for(payload["level"], payload["problem_id"], payload["ref_arch_src"], variant)
    except (ValueError, KeyError, TypeError):
        return "baseline_recipe_request_invalid_or_stale"
    if expected is not None and result.get("baseline_plan") != expected:
        return "baseline_recipe_execution_evidence_missing_or_changed"
    if expected is not None:
        from ..common import fingerprint

        if result.get("reference_sha256") != fingerprint({"source": payload["ref_arch_src"]}):
            return "baseline_recipe_reference_identity_missing_or_changed"
    return None


def artifact_issue(result, variant=None):
    """Replay saved recipes for scorers that have a result, not a task config."""
    if isinstance(result.get("variants"), dict) and variant is None:
        for name, value in result["variants"].items():
            if reason := artifact_issue(value, name):
                return reason
    if result.get("baseline_correctness") is not True:
        return None
    from .baseline_variants import EXTENDED_VARIANTS

    variant = variant or result.get("winning_baseline", result.get("baseline_variant", "eager"))
    if variant in (*EXTENDED_VARIANTS, "neuron_eager", "neuron_compile", "neuron_library"):
        # Neuron's independently validated library portfolio has its own source
        # checks; CUDA preferences and CUDA Graph recipes do not apply there.
        return None
    try:
        from ..common import select_problems

        level, pid = result["baseline_plan"]["task"]
        if type(level) is not int or type(pid) is not int:
            raise ValueError("Invalid baseline task indices")
        source = select_problems(level, problem_ids=[pid])[pid].read_text()
        return issue(result, {"level": level, "problem_id": pid, "ref_arch_src": source,
                              "eval_config": {"baseline_variant": variant}})
    except (KeyError, TypeError, ValueError, OSError, IndexError):
        return "baseline_recipe_artifact_missing_or_invalid"


def baseline_issue(config, baseline, pid):
    from ..engine import task_payload

    if config.device_backend != "cuda":
        return None
    payload = task_payload(config, pid, task="baseline")
    for name, result in baseline.get("variants", {baseline.get("baseline_variant", "eager"): baseline}).items():
        request = {**payload, "eval_config": {**payload["eval_config"], "baseline_variant": name}}
        if reason := issue(result, request):
            return reason
    return None


def framework_rewrite(model, name):
    import torch
    import torch.nn.functional as F

    def acc(value):
        return value if value.dtype == torch.float64 else value.float()

    if name == "channel_rms_norm":
        def forward(x):
            return F.rms_norm(x.movedim(1, -1), (x.shape[1],), eps=model.eps).movedim(-1, 1)
    elif name in {"l1_normalize", "l2_normalize"}:
        def forward(x):
            value = F.normalize(x, p=1 if name == "l1_normalize" else 2, dim=1, eps=0.0)
            return value * x.shape[1] if name == "l1_normalize" else value  # Reference divides by mean(abs(x)).
    elif name == "diagonal_row_scale":
        def forward(diagonal, matrix):
            return diagonal.unsqueeze(-1) * matrix
    elif name == "frobenius_vector_norm":
        def forward(x):
            return x / torch.linalg.vector_norm(x)
    elif name == "var_mean_layer_norm":
        def forward(x):
            layer, work = model.ln, acc(x)
            variance, mean = torch.var_mean(work, tuple(range(-len(layer.normalized_shape), 0)), correction=0, keepdim=True)
            value = (work - mean) * torch.rsqrt(variance + layer.eps)
            if layer.weight is not None:
                value = value * acc(layer.weight)
            if layer.bias is not None:
                value = value + acc(layer.bias)
            return value.to(x.dtype)
    elif name in {"production_layer_norm", "production_var_mean"}:
        def forward(x, weight, bias, eps):
            work = acc(x)
            if name == "production_layer_norm":
                return F.layer_norm(work, (x.shape[-1],), acc(weight), acc(bias), eps).to(x.dtype)
            variance, mean = torch.var_mean(work, -1, correction=0, keepdim=True)
            return ((work - mean) * torch.rsqrt(variance + eps) * acc(weight) + acc(bias)).to(x.dtype)
    elif name == "production_residual_rms":
        def forward(x, residual, weight, eps):
            return F.rms_norm(acc(x) + acc(residual), (x.shape[-1],), acc(weight), eps).to(x.dtype)
    elif name == "production_sdpa":
        def forward(q, k, v, mask):
            return F.scaled_dot_product_attention(acc(q), acc(k), acc(v), attn_mask=mask, dropout_p=0.0).to(q.dtype)
    elif name == "production_gqa":
        def forward(q, k, v):
            return F.scaled_dot_product_attention(acc(q), acc(k), acc(v), enable_gqa=True, dropout_p=0.0).to(q.dtype)
    else:
        raise ValueError(f"Unreviewed framework rewrite: {name}")
    return forward

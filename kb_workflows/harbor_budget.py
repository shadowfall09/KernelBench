"""Read Harbor 0.24 trial accounting without treating absent step fields as zero."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from .common import fingerprint, read_json


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def harbor_budget(directory):
    files = list(Path(directory).glob("*/result.json"))
    result = {"model_calls": None, "candidate_submissions": None, "development_evaluations": None, "profile_requests": None,
              "input_tokens": None, "output_tokens": None, "total_tokens": None, "estimated_cost_usd": None,
              "generation_wall_seconds": None, "development_request_wall_seconds": None, "gpu_execution_seconds": None,
              "models": None, "resource_receipt_verified": False, "billing_cost_certified": False,
              "scope": "Harbor_agent_execution; input_includes_cache; excludes_setup_and_verifier"}
    if len(files) != 1 or files[0].is_symlink():
        result["reason"] = "Missing or ambiguous single-trial result"
        return result
    raw = read_json(files[0])
    if not isinstance(raw, dict):
        result["reason"] = "Malformed Harbor result"
        return result
    contexts = ([raw["agent_result"]] if isinstance(raw.get("agent_result"), dict) else
                [step.get("agent_result") for step in raw.get("step_results") or []])
    # Multi-step omissions are unknown, even if another step has a measurement.
    def known_sum(name):
        values = [ctx.get(name) if isinstance(ctx, dict) else None for ctx in contexts]
        valid = all(finite(value) and value >= 0 and (name == "cost_usd" or type(value) is int) for value in values)
        return sum(values) if values and valid else None
    result.update(input_tokens=known_sum("n_input_tokens"), output_tokens=known_sum("n_output_tokens"),
                  estimated_cost_usd=known_sum("cost_usd"), cache_tokens=known_sum("n_cache_tokens"))
    if result["input_tokens"] is not None and result["output_tokens"] is not None:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    models = sorted({name for ctx in contexts if isinstance(ctx, dict) for name in ctx.get("model_usage") or {}})
    result["models"] = models or None
    timing = raw.get("agent_execution") or {}
    try:
        started, finished = (datetime.fromisoformat(timing[key].replace("Z", "+00:00")) for key in ("started_at", "finished_at"))
        if started.tzinfo is not None and finished.tzinfo is not None:
            elapsed = (finished - started).total_seconds()
            if elapsed >= 0:
                result["generation_wall_seconds"] = elapsed
    except (KeyError, TypeError, ValueError):
        pass
    result["resource_receipt_verified"] = result["total_tokens"] is not None and result["generation_wall_seconds"] is not None
    result["harbor_result_sha256"] = fingerprint(raw)
    return result

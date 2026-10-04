"""Translate former pydra options into the shared command-line interface."""

from __future__ import annotations

import ast


def convert_arguments(arguments):
    converted = []
    aliases = {
        "num_parallel": "parallel",
        "sandbox_timeout": "agent_timeout",
        "eval_timeout": "timeout",
        "aws_profile": "profile",
        "anthropic_model": "model",
        "anthropic_small_model": "small_model",
        "num_samples_per_problem": "num_samples",
        "queue_base_url": "queue_url",
        "eval_mode": "evaluation",
        "use_hidden_tests": "hidden_tests",
    }
    for argument in arguments:
        if "=" not in argument or argument.startswith("--"):
            converted.append(argument)
            continue
        key, value = argument.split("=", 1)
        if key == "dataset_src":
            if value != "local":
                raise ValueError("Verified batch workflows use local references and hidden tests")
            continue
        if key == "subset":
            start, end = ast.literal_eval(value)
            if start is not None:
                converted += ["--start", str(start)]
            if end is not None:
                converted += ["--end", str(end)]
            continue
        if key == "gpu_arch" and value.startswith("["):
            value = ",".join(ast.literal_eval(value))
        if key == "pass_at_k_values":
            converted += ["--pass-at-k", ",".join(str(k) for k in ast.literal_eval(value))]
            continue
        if key == "run_eval":
            if value.lower() in {"false", "0", "no"}:
                converted += ["--stage", "generate"]
            continue
        converted += ["--" + aliases.get(key, key).replace("_", "-"), value]
    return converted

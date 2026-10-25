"""Render self-contained public references and private generators from reviewed sources."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from kb_workflows.benchmarks.production_specs import task
from kb_workflows.common import ROOT


def artifacts():
    from kb_workflows.benchmarks import production_ops

    outputs, rows = {}, []
    dependencies = {"budget_attention": ["attention"], "token_fp8_attention": ["attention"],
                    "budget_moe": ["grouped_moe"], "quantized_moe": ["unpack_int4", "grouped_moe"],
                    "dequant_gemm": ["unpack_int4"]}
    for pid in range(1, 42):
        spec = task(pid)
        names = ["acc", *dependencies.get(spec["operation"], []), spec["operation"]]
        if spec["gradient_indices"]:
            names.append("vjp")
        description = {key: spec[key] for key in ("name", "track", "roles", "upstream_role", "contract_kind", "precision_contract", "fixed_format_comparison", "gradient_indices", "dtype_families")}
        description["dtype_aliases"] = {"fp32": "torch.float32", "bf16": "torch.bfloat16", "fp16": "torch.float16",
                                       "fp8_e4m3fn": "torch.float8_e4m3fn"}
        text = '"""KernelBench Pro production reference. Public contract:\n' + json.dumps(description, indent=2) + '\n"""\n\n'
        implementations = "\n\n".join(inspect.getsource(getattr(production_ops, name)).rstrip() for name in dict.fromkeys(names))
        text += ("import math\n\n" if "math." in implementations else "")
        text += "import torch\nimport torch.nn as nn\n"
        text += "import torch.nn.functional as F\n" if "F." in implementations else ""
        text += "\n\n" + implementations
        call = (f"vjp({spec['operation']}, inputs[:-1], {spec['gradient_indices']}, inputs[-1])"
                if spec["gradient_indices"] else f"{spec['operation']}(*inputs)")
        text += f"\n\n\nclass Model(nn.Module):\n    def forward(self, *inputs):\n        return {call}\n"
        text += f"\n\ndef get_init_inputs():\n    return []\n\n\ndef get_inputs():\n    from kb_workflows.production import make_inputs\n    return make_inputs({pid})\n"
        name = f"{pid}_{spec['name']}.py"
        outputs[Path("KernelBench/level5") / name] = text
        hidden = '"""Private production shape, stride, mask and dynamic-range cases; never prompt material."""\n\n'
        cases = ("hidden_odd", "hidden_strided", "hidden_extreme")
        if spec["family"] in {"normalization", "rmsnorm"}:
            cases += ("hidden_constant",)
        hidden += f"def get_hidden_inputs():\n    from kb_workflows.production import make_inputs\n    return [make_inputs({pid}, case=case) for case in {cases!r}]\n"
        outputs[Path(f"hidden_tests/level5/{pid}_hidden.py")] = hidden
        rows.append({"level": 5, "problem_id": pid, "reference": name,
                     "reference_sha256": hashlib.sha256(text.encode()).hexdigest(), "role": "scored",
                     "layer": spec["layer"], "track": spec["track"], "family": spec["family"]})
    manifest = {"name": "production41", "version": 1, "selection_seed": None, "problems": rows,
                "layer_certificates": False, "reachability_results_collected": False,
                "reference_source": "KernelBench Pro mathematical references; no imported candidate implementations"}
    outputs[Path("kb_workflows/benchmarks/manifests/production41.json")] = json.dumps(manifest, indent=2) + "\n"
    return outputs


def build(root=ROOT, *, check=False):
    outputs = artifacts()
    for name, content in outputs.items():
        path = Path(root) / name
        if check:
            if not path.is_file() or path.read_text() != content:
                raise ValueError(f"Generated production artifact is stale: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    return {"references": 41, "hidden_generators": 41, "manifest": 1, "check_only": check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(check=args.check)))


if __name__ == "__main__":
    main()

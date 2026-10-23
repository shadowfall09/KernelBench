"""Synthetic receipt-schema fixtures for scoring tests; never device evidence."""

import math

from kb_workflows.common import fingerprint
from kb_workflows.operator_cases import PROTOCOL


def attach_operator_evidence(payload, result):
    settings = payload["eval_config"]
    if payload["level"] not in {1, 2, 3} or not settings.get("use_hidden_tests") or result.get("correctness") is not True:
        return result
    spec = settings["operator_inputs"]
    metadata = result.setdefault("metadata", {})
    metadata["reliability_checks"] = [row for row in metadata.get("reliability_checks", []) if not row.get("label", "").startswith("operator/")]
    records = []
    for seed in settings.get("hidden_seeds", (42, 123, 456)):
        for label in spec["labels"]:
            name = f"operator/{seed}/{label}"
            signature = [{"shape": [3, 17], "stride": [34, 2] if label == "strided" else [17, 1], "dtype": "torch.float32"}]
            if payload["level"] in {2, 3}:
                from kb_workflows.model_input_cases import reviewed_spec, shapes

                family = reviewed_spec(payload["level"], payload["problem_id"], payload["ref_arch_src"])
                signature = []
                for shape in shapes(family, label):
                    step = 2 if label == "strided" else 1
                    stride = []
                    for n in reversed(shape):
                        stride.insert(0, step)
                        step *= n
                    signature.append({"shape": list(shape), "stride": stride,
                                      "dtype": {"fp32": "torch.float32", "fp16": "torch.float16", "bf16": "torch.bfloat16"}[settings["precision"]]})
            records.append({"label": name, "family_label": label, "seed": seed, "passed": True,
                            "input_signature": signature, "estimated_input_bytes": 16 * sum(math.prod(row["shape"]) for row in signature)})
            metadata.setdefault("reliability_checks", []).append({"label": name, "seed": seed, "passed": True,
                "input_signature": signature, "orders": ["candidate_first", "reference_first"]})
    metadata["operator_inputs"] = {"protocol": PROTOCOL, "plan_sha256": fingerprint(spec), "cases": records,
        "nominal_timing_changed": False, "reference_parameters_replaced": False}
    return result

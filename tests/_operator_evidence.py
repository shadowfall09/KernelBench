"""Synthetic receipt-schema fixtures for scoring tests; never device evidence."""

from kb_workflows.common import fingerprint
from kb_workflows.operator_cases import PROTOCOL


def attach_operator_evidence(payload, result):
    settings = payload["eval_config"]
    if payload["level"] != 1 or not settings.get("use_hidden_tests") or result.get("correctness") is not True:
        return result
    spec = settings["operator_inputs"]
    metadata = result.setdefault("metadata", {})
    metadata["reliability_checks"] = [row for row in metadata.get("reliability_checks", []) if not row.get("label", "").startswith("operator/")]
    records = []
    for seed in settings.get("hidden_seeds", (42, 123, 456)):
        for label in spec["labels"]:
            name = f"operator/{seed}/{label}"
            signature = [{"shape": [3, 17], "stride": [34, 2] if label == "strided" else [17, 1], "dtype": "torch.float32"}]
            records.append({"label": name, "family_label": label, "seed": seed, "passed": True,
                            "input_signature": signature, "estimated_input_bytes": 816})
            metadata.setdefault("reliability_checks", []).append({"label": name, "seed": seed, "passed": True,
                "input_signature": signature, "orders": ["candidate_first", "reference_first"]})
    metadata["operator_inputs"] = {"protocol": PROTOCOL, "plan_sha256": fingerprint(spec), "cases": records,
        "nominal_timing_changed": False, "reference_parameters_replaced": False}
    return result

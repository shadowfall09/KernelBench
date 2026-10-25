"""Lightweight contract lookup and coverage preflight, without importing Torch."""

import hashlib
import json
from pathlib import Path

CONTRACTS = Path(__file__).with_name("verification_contracts.json")


def contract_for(level, pid, reference_source):
    contract = json.loads(CONTRACTS.read_text())["tasks"].get(f"{level}/{pid}")
    if contract is not None and contract["reference_sha256"] != hashlib.sha256(reference_source.encode()).hexdigest():
        raise ValueError("Strict contract reference snapshot changed; review its domain and oracle")
    return contract


def coverage(level, selected):
    available = json.loads(CONTRACTS.read_text())["tasks"]
    supported, unsupported = [], []
    for pid, path in selected.items():
        contract = available.get(f"{level}/{pid}")
        if contract is None:
            unsupported.append(pid)
        else:
            if contract["reference_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Strict contract reference snapshot changed for {level}/{pid}")
            supported.append(pid)
    return {"supported": supported, "unsupported": unsupported,
            "precision_scope": "reviewed_auxiliary_cases; original_timing_inputs_keep_native_comparator"}

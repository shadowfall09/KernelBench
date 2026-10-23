"""Frozen L1 oracle selection and error policies; imports no tensor SDK."""

import hashlib
import json
from pathlib import Path

PROTOCOL = "kernelbench_pro_operator_oracles_v1"
MANIFEST = Path(__file__).with_name("operator_oracles.json")


def spec_for(level, pid, source):
    document = json.loads(MANIFEST.read_text())
    if document.get("protocol") != PROTOCOL:
        raise ValueError("Unknown independent operator oracle manifest")
    result = document["tasks"].get(f"{level}/{pid}")
    if result is None:
        raise ValueError("No independent operator oracle adapter for this task")
    if result["reference_sha256"] != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Independent operator oracle reference changed; review the formula and domain")
    if result.get("family") == "composite":
        from .composite_specs import program_for

        program_for(result)
    return {**{key: document[key] for key in ("oracle", "precision", "scope", "native_input_roles_preserved")}, **result}

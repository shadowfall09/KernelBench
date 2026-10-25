"""Shared independent-oracle preparation for reference calibration and timing."""

from __future__ import annotations


def prepare(reference, inputs, *, spec, problem_id):
    if spec.get("family") == "composite":
        from .composite_oracles import Prepared

        return Prepared(reference, inputs, expected_program=spec["program_sha256"])
    if spec.get("family") == "model_block":
        from .model_oracles import Prepared

        return Prepared(reference, inputs, problem_id=problem_id, expected_program=spec["program_sha256"])
    from .operator_oracles import estimate_bytes, execute

    class Operator:
        estimated_host_bytes = estimate_bytes(problem_id, reference, inputs)

        def execute(self):
            return execute(problem_id, reference, inputs)

        def receipt(self):
            return None

    return Operator()

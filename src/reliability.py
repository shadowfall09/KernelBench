"""Research-informed correctness checks; sources and limits: docs/RELIABILITY.md.

Independently implemented from the evaluation ideas in robust-kbench,
KernelBench-Hard and KernelBenchX. This module is not a security sandbox.
"""

from pathlib import Path

import torch

from .validation import clone_tensors, compare_outputs, preserve_backend_settings, synchronize

PROTOCOL = "paired_order_multiseed_stress_v1"
HIDDEN_SEEDS = (42, 123, 456)
# Only these reviewed Level 1 tasks accept arbitrary finite real values in input 0.
# Excludes symmetric/triangular matrices, probability targets, indices and masks.
STRESS_TASKS = frozenset((*range(1, 13), *range(16, 28), 47, 48, 49, 51, 52))


def stress_enabled(hidden_path):
    if hidden_path is None:
        return False
    path = Path(hidden_path).resolve()
    trusted = Path(__file__).resolve().parents[1] / "hidden_tests" / "level1"
    return path.parent == trusted and path.stem in {f"{pid}_hidden" for pid in STRESS_TASKS}


def stress_cases(inputs, seed):
    """Yield legal value variants of input 0 without changing layout or other args."""
    value = inputs[0]
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not value.numel():
        raise ValueError("Reviewed stress input 0 must be a nonempty floating tensor")
    for name in ("zero", "scaled_32", "sparse_outliers"):
        changed = clone_tensors(value)
        if name == "zero":
            changed.zero_()
        elif name == "scaled_32":
            changed.mul_(32)
        else:
            # Chunking bounds temporary mask storage even for very large inputs.
            generator = torch.Generator(device=value.device).manual_seed(seed)
            flat = changed.reshape(-1)
            injected = 0
            for start in range(0, flat.numel(), 1 << 20):
                chunk = flat[start:start + (1 << 20)]
                mask = torch.rand(chunk.shape, device=value.device, generator=generator) < 0.001
                count = int(mask.sum().item())
                chunk[mask] = (torch.randn(count, device=value.device, generator=generator) * 50).to(value.dtype)
                injected += count
            if not injected:
                # Small tensors must exercise at least one actual outlier.
                flat[0] = 50
            changed.copy_(flat.reshape(value.shape))
        yield name, [changed, *inputs[1:]]


def check_paired_calls(reference, candidate, inputs, *, seed, tolerance, device,
                       set_seed, metadata, label, record_first_call=False, clock=None, verify_pair=None):
    """Compare C→R and R→C pairs, advancing both models equally, with RNG replay.

    Snapshot each output before the other implementation can reuse its storage.
    Pairing two reference calls also supports stateful modules such as BatchNorm.
    """
    from kb_workflows.determinism import observe
    from kb_workflows.state_audit import observe as observe_state

    observe(reference, candidate, inputs, seed=seed, device=device, set_seed=set_seed, tolerance=tolerance, label=label)
    observe_state(reference, candidate, inputs, seed=seed, device=device, set_seed=set_seed, tolerance=tolerance, label=label)
    receipt = {"label": label, "seed": seed, "passed": False, "orders": []}
    metadata.setdefault("reliability_checks", []).append(receipt)
    metadata["reliability_protocol"] = PROTOCOL
    for order in ("candidate_first", "reference_first"):
        outputs = {}
        names = ("candidate", "reference") if order == "candidate_first" else ("reference", "candidate")
        for name in names:
            try:
                fresh = clone_tensors(inputs)
                set_seed(seed)
                synchronize(device)
                started = clock() if record_first_call and name == "candidate" else None
                with preserve_backend_settings():
                    output = (candidate if name == "candidate" else reference)(*fresh)
                    synchronize(device)
                if started is not None:
                    metadata["first_forward_ms"] = (clock() - started) * 1000
                    metadata["cold_start_ms"] = metadata.get("candidate_setup_ms", 0) + metadata["first_forward_ms"]
                    record_first_call = False
                outputs[name] = clone_tensors(output)
                synchronize(device)
            except Exception as exc:
                receipt.update(failure_kind=f"{name}_error", issue=f"{type(exc).__name__}: {exc}")
                return f"{label}/{order}: {receipt['failure_kind']}: {receipt['issue']}"
        issue = compare_outputs(outputs["reference"], outputs["candidate"], tolerance)
        receipt["orders"].append(order)
        if issue:
            receipt.update(failure_kind="output_mismatch", issue=issue)
            return f"{label}/{order}: {issue}"
        if verify_pair:
            verify_pair(inputs, outputs)
    receipt["passed"] = True
    return None

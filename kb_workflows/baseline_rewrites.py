"""Reviewed framework-only equivalent rewrites, tied to reference source hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("baseline_rewrites.json")


def rewrite_for(level, pid, source):
    item = json.loads(MANIFEST.read_text()).get(f"{level}/{pid}")
    if item and item["reference_sha256"] != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Baseline rewrite reference changed; review semantic equivalence")
    return item["rewrite"] if item else None


def framework_rewrite(model, name):
    import torch.nn.functional as F

    if name == "channel_rms_norm":
        def forward(x):
            return F.rms_norm(x.movedim(1, -1), (x.shape[1],), eps=model.eps).movedim(-1, 1)
    elif name in {"l1_normalize", "l2_normalize"}:
        def forward(x):
            value = F.normalize(x, p=1 if name == "l1_normalize" else 2, dim=1, eps=0.0)
            return value * x.shape[1] if name == "l1_normalize" else value  # Reference divides by mean(abs(x)).
    else:
        raise ValueError(f"Unreviewed framework rewrite: {name}")
    return forward

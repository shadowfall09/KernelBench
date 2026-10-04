#!/usr/bin/env python3
"""Generate TF32/FP16/BF16 baselines, with optional torch.compile, on any verifier backend."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb_workflows.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--stage", "baseline", *sys.argv[1:]]))

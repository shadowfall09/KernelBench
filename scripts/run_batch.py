#!/usr/bin/env python3
"""Unified Verified workflow; see docs/BATCH_WORKFLOWS.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb_workflows.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

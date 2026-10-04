#!/usr/bin/env python3
"""Compatibility entrypoint: queue evaluation now uses the shared Verified workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb_workflows.cli import main
from kb_workflows.legacy import convert_arguments

if __name__ == "__main__":
    raise SystemExit(main(["--stage", "evaluate", "--evaluation", "queue", *convert_arguments(sys.argv[1:])]))

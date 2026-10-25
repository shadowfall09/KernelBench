"""Freeze an existing reference calibration ledger without running experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kb_workflows.benchmarks.reference_population import build
from kb_workflows.common import atomic_json
from kb_workflows.config import parse_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--write", action="store_true", help="Write the new manifest; no device or model execution")
    args = parser.parse_args(argv)
    try:
        config = parse_config(["--config", str(args.config), "--stage", "analyze", "--dry-run"])
        if config.reference_population:
            raise ValueError("Freeze from the original calibration configuration, without an existing reference population")
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Reference population output must be a new path")
        manifest = build(config)
        if args.write:
            atomic_json(args.output, manifest)
        print(json.dumps({"manifest": manifest, "output": str(args.output.resolve()), "written": args.write,
                          "experiments_started": False, "candidate_executed": False, "timings_collected": False}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Export a reachability witness from completed independent verification, without running it."""

from __future__ import annotations

import argparse
from pathlib import Path

from kb_workflows.common import atomic_json, read_json
from kb_workflows.config import RunConfig
from kb_workflows.headroom import witness_from_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--problem", type=int, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = RunConfig(**read_json(args.run / "run_config.json"))
    config.runs_dir, config.run_name = str(args.run.resolve().parent), args.run.name
    witness = witness_from_run(config, args.problem, args.sample, read_json(args.calibration))
    atomic_json(args.output, witness)
    print(args.output)


if __name__ == "__main__":
    main()

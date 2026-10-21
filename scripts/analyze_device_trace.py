"""Analyze an existing device trace; this command does not execute a workload."""

from __future__ import annotations

import argparse
from pathlib import Path

from kb_workflows.common import atomic_json, fingerprint, read_json
from kb_workflows.performance import timeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    trace = read_json(args.trace)
    result = timeline(trace, device=args.device)
    result["trace_sha256"] = fingerprint(trace)
    atomic_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()

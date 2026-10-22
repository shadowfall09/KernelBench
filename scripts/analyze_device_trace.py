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
    parser.add_argument("--backend", choices=("cuda", "neuron"), default="cuda")
    parser.add_argument("--timestamp-unit", choices=("ns", "us", "ms"))
    parser.add_argument("--window", type=float, nargs=2)
    args = parser.parse_args(argv)
    trace = read_json(args.trace)
    if args.backend == "neuron":
        from kb_workflows.neuron_profile import timeline as neuron_timeline

        result = neuron_timeline(trace, timestamp_unit=args.timestamp_unit, window=args.window)
    else:
        if args.timestamp_unit or args.window:
            raise ValueError("CUDA Chrome traces carry their own marker and microsecond units")
        result = timeline(trace, device=args.device)
    result["trace_sha256"] = fingerprint(trace)
    atomic_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()

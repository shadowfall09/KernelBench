"""Read existing Neuron Explorer Parquet tables; never execute a workload/tool."""

from __future__ import annotations

import argparse
from pathlib import Path

from kb_workflows.common import atomic_json
from kb_workflows.neuron_tables import load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--max-file-mb", type=int, default=64)
    parser.add_argument("--max-decoded-mb", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Neuron work output must be a new path")
        report = load(args.profile, max_rows=args.max_rows, max_file_bytes=args.max_file_mb * 1024**2,
                      max_decoded_bytes=args.max_decoded_mb * 1024**2)
        atomic_json(args.output, report)
        print(args.output)
        return 0
    except (ValueError, OSError, TypeError, KeyError, ImportError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

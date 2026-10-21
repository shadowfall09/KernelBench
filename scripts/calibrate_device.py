"""Explicit opt-in calibration command; never invoked automatically by evaluation."""

from __future__ import annotations

import argparse

from kb_workflows.calibration import CUDADevice
from kb_workflows.common import atomic_json


def main(argv=None):
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--matrix-sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args(argv)
    result = CUDADevice(args.device).calibrate(matrix_sizes=args.matrix_sizes,
        memory_bytes=args.memory_mib * 1024**2, trials=args.trials)
    atomic_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()

"""Read existing Neuron Explorer Parquet tables; never execute a workload/tool."""

from __future__ import annotations

import argparse
from pathlib import Path

from kb_workflows.common import atomic_json, read_json
from kb_workflows.neuron_tables import load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--max-file-mb", type=int, default=64)
    parser.add_argument("--max-decoded-mb", type=int, default=256)
    parser.add_argument("--capture", type=Path, help="Existing verifier capture receipt or profile result")
    parser.add_argument("--review", type=Path, help="Trusted review pairing this report and capture")
    parser.add_argument("--calibration", type=Path, help="Existing calibration from the capture device")
    parser.add_argument("--processing", type=Path, help="Existing verified Explorer processing receipt or result")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Neuron work output must be a new path")
        requested = (args.capture, args.review, args.calibration)
        if any(requested) and not all(requested):
            raise ValueError("Neuron model requires capture, review and calibration together")
        if args.processing and not all(requested):
            raise ValueError("Processing evidence requires capture, review and calibration")
        report = load(args.profile, max_rows=args.max_rows, max_file_bytes=args.max_file_mb * 1024**2,
                      max_decoded_bytes=args.max_decoded_mb * 1024**2)
        if all(requested):
            from kb_workflows.neuron_work import execution_model

            try:
                capture = read_json(args.capture)
                capture = capture.get("capture_receipt", capture)
                processing = read_json(args.processing) if args.processing else None
                if args.processing:
                    if not isinstance(processing, dict):
                        raise ValueError("Requested processing receipt is missing or invalid; no manual fallback")
                    processing = processing.get("processing", processing)
                    if not isinstance(processing, dict):
                        raise ValueError("Requested processing result has no complete receipt; no manual fallback")
                report = execution_model(report, capture, read_json(args.review), read_json(args.calibration), processing=processing)
            except (ValueError, OSError, TypeError, KeyError, AttributeError, OverflowError) as exc:
                atomic_json(args.output, {"report": report, "execution_model": {"status": "not_measured", "reason": str(exc)}})
                print(f"Error: {exc}; raw work report retained at {args.output}")
                return 2
        atomic_json(args.output, report)
        print(args.output)
        return 0
    except (ValueError, OSError, TypeError, KeyError, ImportError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

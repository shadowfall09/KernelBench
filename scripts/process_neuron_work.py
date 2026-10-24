"""Preview or explicitly process existing Neuron artifacts; never run a workload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from kb_workflows.common import atomic_json, read_json
from kb_workflows.neuron_processing import DEFAULT_LIMITS, DEFAULT_TABLE_LIMITS, core_valid, process_capture
from kb_workflows.neuron_work import validate_capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--binary", default="neuron-explorer")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-artifact-file-mb", type=int, default=128)
    parser.add_argument("--max-artifact-mb", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--max-table-file-mb", type=int, default=64)
    parser.add_argument("--max-decoded-mb", type=int, default=256)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        capture = read_json(args.capture)
        capture = capture.get("capture_receipt", capture)
        validate_capture(capture)
        core_valid(capture)
        if capture.get("artifact_file_bytes") is None:
            raise ValueError("Automatic processing requires the native capture byte inventory")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("Offline processing timeout must be positive and finite")
        directory = args.work_dir or args.output.with_name(args.output.stem + "-processing")
        if args.output.exists() or args.output.is_symlink() or directory.exists() or directory.is_symlink():
            raise ValueError("Processing output and work directory must be new paths")
        if directory.resolve().is_relative_to(args.output.resolve()) or args.output.resolve().is_relative_to(
            args.artifacts.resolve()
        ):
            raise ValueError("Processing output must be outside the capture and cannot contain the work directory")
        limits = {
            **DEFAULT_LIMITS,
            "max_file_bytes": args.max_artifact_file_mb * 1024**2,
            "max_bytes": args.max_artifact_mb * 1024**2,
        }
        from kb_workflows.neuron_processing import limits_valid

        limits_valid(limits)
        table_limits = {
            "max_rows": args.max_rows,
            "max_file_bytes": args.max_table_file_mb * 1024**2,
            "max_decoded_bytes": args.max_decoded_mb * 1024**2,
        }
        if set(table_limits) != set(DEFAULT_TABLE_LIMITS) or any(v <= 0 for v in table_limits.values()):
            raise ValueError("Table processing row/file/decoded limits must be positive")
        preview = {
            "operation": "offline_Explorer_view",
            "capture_receipt_sha256": capture["receipt_sha256"],
            "source_directory": str(args.artifacts.resolve()),
            "work_directory": str(directory.resolve()),
            "output": str(args.output.resolve()),
            "binary": args.binary,
            "timeout": args.timeout,
            "artifact_limits": limits,
            "table_limits": table_limits,
            "tool_invoked": False,
            "device_workload_started": False,
        }
        if not args.execute:
            print(json.dumps(preview, indent=2))
            return 0
        result = process_capture(
            capture,
            args.artifacts,
            directory=directory,
            binary=args.binary,
            timeout=args.timeout,
            limits=limits,
            table_limits=table_limits,
        )
        atomic_json(args.output, result)
        print(
            json.dumps(
                {
                    **preview,
                    "tool_invocation_requested": True,
                    "tool_invoked": result["tool_invoked"],
                    "status": result["status"],
                },
                indent=2,
            )
        )
        return 0 if result["status"] == "ok" else 2
    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

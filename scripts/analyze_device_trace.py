"""Inspect device traces, Neuron tables, or offline profile processing."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from kb_workflows.analysis.performance import native_timeline, timeline
from kb_workflows.common import atomic_json, fingerprint, read_json
from kb_workflows.hardware.neuron_processing import DEFAULT_LIMITS, DEFAULT_TABLE_LIMITS, core_valid, process_capture
from kb_workflows.hardware.neuron_tables import load
from kb_workflows.hardware.neuron_work import validate_capture


def timeline_main(argv=None):
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
        from kb_workflows.hardware.neuron_tables import timeline as neuron_timeline

        result = neuron_timeline(trace, timestamp_unit=args.timestamp_unit, window=args.window)
    else:
        if args.timestamp_unit or args.window:
            raise ValueError("CUDA Chrome traces carry their own marker and microsecond units")
        result = timeline(trace, device=args.device)
    result["trace_sha256"] = fingerprint(trace)
    atomic_json(args.output, result)
    print(args.output)
    return 0


def native_main(argv=None):
    parser = argparse.ArgumentParser(description="Replay an existing native origin report; never start a collector or workload.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from kb_workflows.provenance.origin_task import MAX_REPORT_BYTES

        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Native timeline output must be a new path")
        if args.report.is_symlink() or not args.report.is_file() or args.report.stat().st_size > MAX_REPORT_BYTES:
            raise ValueError("Native origin report must be a bounded regular file")
        with args.report.open("rb") as stream:
            raw = stream.read(MAX_REPORT_BYTES + 1)
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("Native origin report exceeds the analysis byte limit")
        result = native_timeline(json.loads(raw))
        atomic_json(args.output, result)
        print(args.output)
        return 0 if result["status"] == "measured" else 2
    except (ValueError, OSError, TypeError, KeyError, AttributeError, ImportError) as exc:
        print(f"Error: {exc}")
        return 2


def neuron_tables_main(argv=None):
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
            from kb_workflows.hardware.neuron_work import execution_model

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

def neuron_process_main(argv=None):
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
        from kb_workflows.hardware.neuron_processing import limits_valid

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


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"timeline": timeline_main, "native": native_main, "neuron-tables": neuron_tables_main, "neuron-process": neuron_process_main}
    if not args or args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s {timeline,native,neuron-tables,neuron-process} ...")
        parser.epilog = "Use COMMAND --help for command-specific options. Legacy TRACE --backend ... remains a timeline."
        parser.print_help()
        return 0
    return commands[args.pop(0)](args) if args[0] in commands else timeline_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Verifier-owned offline Explorer view: no capture/workload/device is started."""

from __future__ import annotations

import base64
import hashlib
import math
import shutil
import stat
import time
from pathlib import Path, PurePosixPath

from ..common import evaluator_digest, fingerprint
from ..provenance.work_counters import sha
from .neuron_tables import file_digest, integer
from .neuron_work import seal, validate_capture

PROTOCOL = "kernelbench_pro_neuron_processing_v1"
COPY_POLICY = "complete_capture_inventory; regular_files; byte_verified_readonly_snapshot"
CORE_SCOPE = "only_visible_physical_core_of_verified_capture; no_numeric_index_translation"
DEFAULT_LIMITS = {"max_files": 4096, "max_file_bytes": 128 * 1024**2, "max_bytes": 512 * 1024**2}
DEFAULT_TABLE_LIMITS = {"max_rows": 200_000, "max_file_bytes": 64 * 1024**2, "max_decoded_bytes": 256 * 1024**2}


def command(binary, source, target):
    return [
        str(binary),
        "view",
        "--session-dir",
        str(source),
        "--output-format",
        "parquet",
        "--output-file",
        str(target),
    ]


def limits_valid(limits):
    if (
        not isinstance(limits, dict)
        or set(limits) != set(DEFAULT_LIMITS)
        or any(type(v) is not int or v <= 0 for v in limits.values())
        or limits["max_file_bytes"] > limits["max_bytes"]
    ):
        raise ValueError("Neuron artifact limits must be positive with per-file size within total size")


def file_manifest(directory, limits, *, checkpoint=lambda: None):
    """Admit the complete local tree before hashing any regular file."""
    limits_valid(limits)
    if Path(directory).is_symlink():
        raise ValueError("Neuron artifact inventory root cannot be a symlink")
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Neuron artifacts must be a local directory")
    pending, total, entries = [], 0, 0
    for path in root.rglob("*"):
        checkpoint()
        entries += 1
        if entries > limits["max_files"] * 2:
            raise ValueError("Neuron artifact directory exceeds its entry envelope")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Neuron artifacts cannot contain links or special files")
        size = path.stat().st_size
        total += size
        if len(pending) >= limits["max_files"] or size > limits["max_file_bytes"] or total > limits["max_bytes"]:
            raise ValueError("Complete Neuron artifact inventory exceeds its file/byte envelope")
        pending.append((path, size))
    result = {}
    for path, size in sorted(pending):
        checkpoint()
        result[path.relative_to(root).as_posix()] = {"sha256": file_digest(path), "bytes": size}
        if path.is_symlink() or path.stat().st_size != size:
            raise ValueError("Neuron artifacts changed during inventory")
    return result


def source_valid(files, capture):
    if (
        {name: row["sha256"] for name, row in files.items()} != capture["artifact_files"]
        or {name: row["bytes"] for name, row in files.items()} != capture.get("artifact_file_bytes")
    ) or any(row["bytes"] <= 0 for name, row in files.items() if PurePosixPath(name).suffix in {".neff", ".ntff"}):
        raise ValueError("Native files differ from the complete verified capture inventory")


def snapshot(source, target, files, *, checkpoint):
    target.mkdir()
    for name, entry in files.items():
        checkpoint()
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        observed, count = hashlib.sha256(), 0
        with (Path(source) / name).open("rb") as src, path.open("xb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                checkpoint()
                count += len(block)
                if count > entry["bytes"]:
                    raise ValueError("Native artifact grew during snapshot")
                observed.update(block)
                dst.write(block)
        if count != entry["bytes"] or observed.hexdigest() != entry["sha256"]:
            raise ValueError("Native artifact changed while copying its snapshot")
        path.chmod(0o444)


def completed(result):
    return (
        isinstance(result, dict)
        and type(result.get("exit_code")) is int
        and result["exit_code"] == 0
        and result.get("timed_out") is False
        and type(result.get("request_wall_seconds")) in {int, float}
        and math.isfinite(result["request_wall_seconds"])
        and result["request_wall_seconds"] >= 0
    )


def tool_identity(binary, run, directory, remaining):
    from ..evaluation.tool_runner import environment

    located = shutil.which(str(binary))
    if located is None:
        raise ValueError("Configured neuron-explorer executable is unavailable")
    path = Path(located).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 256 * 1024**2:
        raise ValueError("Neuron Explorer must be a bounded regular executable")
    before = file_digest(path)
    log = directory / "explorer-version.log"
    result = run([str(path), "--version"], min(5, remaining()), log, environment=environment())
    if not completed(result) or not log.is_file() or log.is_symlink() or not 0 < log.stat().st_size <= 1024**2:
        raise ValueError("Neuron Explorer version discovery failed or timed out")
    version = log.read_text(errors="strict")
    if not version.strip() or file_digest(path) != before:
        raise ValueError("Neuron Explorer version is empty or its executable changed")
    return {
        "name": "neuron-explorer",
        "path": str(path),
        "sha256": before,
        "version": version,
        "version_sha256": hashlib.sha256(version.encode()).hexdigest(),
        "binary_unchanged": True,
    }


def core_valid(capture):
    physical = capture["device"]["physical_device"]["physical_core"]
    allocation = capture.get("neuron_core_allocation")
    if (
        not isinstance(allocation, list)
        or len(allocation) != 1
        or type(allocation[0]) is not int
        or allocation != [physical]
    ):
        raise ValueError("Automatic processing requires exactly the verified physical core to have been visible")


def _process(capture, source, directory, *, binary, run, simulated, timeout, limits, table_limits):
    from ..evaluation.tool_runner import environment
    from .neuron_tables import load

    deadline = time.monotonic() + timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise ValueError("Offline Neuron processing exhausted its request budget")
        return value

    files = file_manifest(source, limits, checkpoint=remaining)
    source_valid(files, capture)
    frozen, output = directory / "input", directory / "parquet"
    snapshot(source, frozen, files, checkpoint=remaining)
    tool = tool_identity(binary, run, directory, remaining)
    argv = command(tool["path"], frozen, output)
    process = run(argv, remaining(), directory / "explorer.log", environment=environment())
    if not completed(process):
        raise ValueError("Neuron Explorer view failed or timed out; no partial association")
    if file_digest(tool["path"]) != tool["sha256"]:
        raise ValueError("Neuron Explorer executable changed during processing")
    if (
        file_manifest(source, limits, checkpoint=remaining) != files
        or file_manifest(frozen, limits, checkpoint=remaining) != files
    ):
        raise ValueError("Native artifacts or their snapshot changed during processing")
    outputs = file_manifest(output, limits, checkpoint=remaining)
    report = load(output, **table_limits)
    remaining()
    if file_manifest(output, limits, checkpoint=remaining) != outputs:
        raise ValueError("Processed Neuron files changed while reading their complete tables")
    if (
        file_manifest(source, limits, checkpoint=remaining) != files
        or file_manifest(frozen, limits, checkpoint=remaining) != files
        or file_digest(tool["path"]) != tool["sha256"]
    ):
        raise ValueError("Neuron inputs or executable changed before report decoding completed")
    remaining()
    receipt = seal(
        {
            "protocol": PROTOCOL,
            "evaluator_sha256": evaluator_digest(),
            "tool": tool,
            "command": argv,
            "capture_receipt_sha256": capture["receipt_sha256"],
            "report_receipt_sha256": report["receipt_sha256"],
            "source_files": files,
            "output_files": outputs,
            "report_files": report["files"],
            "copy_policy": COPY_POLICY,
            "core_binding_scope": CORE_SCOPE,
            "artifact_limits": limits,
            "table_limits": table_limits,
            "source_after_processing_unchanged": True,
            "snapshot_after_processing_unchanged": True,
            "output_after_decoding_unchanged": True,
            "process": {key: process[key] for key in ("exit_code", "timed_out", "request_wall_seconds")},
            "simulated": simulated,
            "device_workload_started": False,
            "formal_origin_qualified": False,
            "scored_wall_times_replaced": False,
        }
    )
    validate(receipt, report, capture, allow_simulated=simulated)
    return {
        "status": "ok",
        "report": report,
        "processing": receipt,
        "work_directory": str(directory),
        "device_workload_started": False,
        "scored_wall_times_replaced": False,
    }


def failure_logs(directory):
    result = {}
    for name in ("explorer-version.log", "explorer.log"):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(64 * 1024)
        result[name] = {
            "bytes": size,
            "prefix_base64": base64.b64encode(raw).decode(),
            "prefix_sha256": hashlib.sha256(raw).hexdigest(),
            "complete": len(raw) == size,
        }
    return result


def process_capture(
    capture, source, *, directory, binary="neuron-explorer", timeout=120, limits=None, table_limits=None, run=None
):
    """Explicit offline execution. The CLI previews by default; work files survive."""
    from ..evaluation.tool_runner import process

    if Path(directory).is_symlink():
        raise ValueError("Neuron processing requires a new work directory, not a symlink")
    directory = Path(directory).resolve()
    if directory.exists() or directory.is_symlink():
        raise ValueError("Neuron processing requires a new work directory")
    simulated = run is not None or not isinstance(capture, dict) or capture.get("simulated") is not False
    invoked = False

    def tracked_run(*args, **kwargs):
        nonlocal invoked
        invoked = True
        return (run or process)(*args, **kwargs)

    try:
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Neuron processing timeout must be positive and finite")
        limits = dict(DEFAULT_LIMITS if limits is None else limits)
        limits_valid(limits)
        table_limits = {**DEFAULT_TABLE_LIMITS, **(table_limits or {})}
        if set(table_limits) != set(DEFAULT_TABLE_LIMITS) or any(
            type(v) is not int or v <= 0 for v in table_limits.values()
        ):
            raise ValueError("Neuron table-processing limits must be positive known row/file/decoded estimates")
        validate_capture(capture, allow_simulated=run is not None)
        core_valid(capture)
        source = Path(source).resolve(strict=True)
        if source.is_relative_to(directory) or directory.is_relative_to(source):
            raise ValueError("Neuron processing work directory must be separate from the native capture")
        directory.mkdir(parents=True, mode=0o700)
        result = _process(
            capture,
            source,
            directory,
            binary=binary,
            run=tracked_run,
            simulated=simulated,
            timeout=timeout,
            limits=limits,
            table_limits=table_limits,
        )
        return {**result, "tool_invoked": invoked}
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, UnicodeError, ImportError) as exc:
        return {
            "status": "not_measured",
            "reason": str(exc),
            "failure_artifacts": failure_logs(directory),
            "work_directory": str(directory),
            "device_workload_started": False,
            "scored_wall_times_replaced": False,
            "tool_invoked": invoked,
        }


def descriptors_valid(files, limits):
    if not isinstance(files, dict) or not files or len(files) > limits["max_files"]:
        raise ValueError("Processing receipt has no complete bounded artifact inventory")
    for name, entry in files.items():
        path = PurePosixPath(name) if isinstance(name, str) else None
        if (
            path is None
            or not name
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "bytes"}
            or not sha(entry.get("sha256"))
            or not integer(entry.get("bytes"))
            or entry["bytes"] > limits["max_file_bytes"]
        ):
            raise ValueError("Invalid processed artifact descriptor")
    if sum(row["bytes"] for row in files.values()) > limits["max_bytes"]:
        raise ValueError("Processing artifact inventory exceeds its total byte envelope")


def validate(value, report, capture, *, allow_simulated=False):
    from .neuron_tables import validate as validate_report

    validate_capture(capture, allow_simulated=allow_simulated)
    validate_report(report)
    core_valid(capture)
    if (
        not isinstance(value, dict)
        or value.get("protocol") != PROTOCOL
        or value.get("evaluator_sha256") != evaluator_digest()
        or value.get("receipt_sha256") != fingerprint({k: v for k, v in value.items() if k != "receipt_sha256"})
        or (value.get("simulated") is not False and not allow_simulated)
        or value.get("capture_receipt_sha256") != capture["receipt_sha256"]
        or value.get("report_receipt_sha256") != report["receipt_sha256"]
        or value.get("report_files") != report["files"]
        or value.get("copy_policy") != COPY_POLICY
        or value.get("core_binding_scope") != CORE_SCOPE
        or value.get("source_after_processing_unchanged") is not True
        or value.get("snapshot_after_processing_unchanged") is not True
        or value.get("output_after_decoding_unchanged") is not True
        or not completed(value.get("process"))
        or value.get("device_workload_started") is not False
        or value.get("formal_origin_qualified") is not False
        or value.get("scored_wall_times_replaced") is not False
    ):
        raise ValueError("Neuron processing receipt is missing, stale, changed, incomplete or simulated")
    limits = value.get("artifact_limits")
    limits_valid(limits)
    table_limits = value.get("table_limits")
    if (
        not isinstance(table_limits, dict)
        or set(table_limits) != set(DEFAULT_TABLE_LIMITS)
        or any(type(v) is not int or v <= 0 for v in table_limits.values())
        or table_limits != report.get("resource_envelope")
        or sum(row["rows"] for row in report["files"].values()) > table_limits["max_rows"]
        or any(row["bytes"] > table_limits["max_file_bytes"] for row in report["files"].values())
    ):
        raise ValueError("Processing table envelopes differ from the complete decoded report")
    for name in ("source_files", "output_files"):
        descriptors_valid(value.get(name), limits)
    source_valid(value["source_files"], capture)
    if any(
        value["output_files"].get(name) != {key: entry[key] for key in ("sha256", "bytes")}
        for name, entry in report["files"].items()
    ):
        raise ValueError("Processed tables differ from the complete output file inventory")
    tool, argv = value.get("tool"), value.get("command")
    if (
        not isinstance(tool, dict)
        or tool.get("name") != "neuron-explorer"
        or not sha(tool.get("sha256"))
        or not isinstance(tool.get("version"), str)
        or not tool["version"].strip()
        or tool.get("version_sha256") != hashlib.sha256(tool["version"].encode()).hexdigest()
        or tool.get("binary_unchanged") is not True
        or not isinstance(argv, list)
        or len(argv) != 8
        or any(not isinstance(part, str) for part in argv)
        or argv[0] != tool.get("path")
        or argv[1:3] != ["view", "--session-dir"]
        or argv[4:7] != ["--output-format", "parquet", "--output-file"]
    ):
        raise ValueError("Neuron processing lacks the exact completed Explorer view command and tool identity")
    binary, source, target = (PurePosixPath(argv[i]) for i in (0, 3, 7))
    if (
        any(not path.is_absolute() or ".." in path.parts for path in (binary, source, target))
        or source.name != "input"
        or target.name != "parquet"
        or source.parent != target.parent
    ):
        raise ValueError("Neuron processing command used different input or output paths")
    return value

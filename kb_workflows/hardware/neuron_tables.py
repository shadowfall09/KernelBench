"""Offline Neuron Explorer 12.0.1 tables; no SDK/tool/device is started here."""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from pathlib import Path

from ..analysis.calibration import positive
from ..analysis.performance import interval_union
from ..common import evaluator_digest, fingerprint

PROTOCOL = "kernelbench_pro_neuron_explorer_work_v1"
SCHEMA = "12.0.1"
TABLES = {
    "Metadata": ("profile_schema_version", "capture_type", "num_physical_cores", "vnc_enabled", "explorer_version",
                 "neuron_device_idx", "neuroncore_idx", "first_hw_timestamp", "last_hw_timestamp"),
    "ExecutionInfo": ("execution_index", "neff_name", "execution_start_ts", "execution_end_ts"),
    "Instruction": ("id", "execution_index", "start_ts", "end_ts", "duration_ns", "engine", "engine_instance_idx", "execution_unit",
                    "pcore_idx", "opcode", "pc", "compiler_pc", "nki_call_site_id", "raw_flops", "adjusted_flops",
                    "hbm_read_bytes", "hbm_write_bytes", "src1_dtype", "src2_dtype", "dst_dtype", "tensor_instruction_type"),
    "DmaPacket": ("id", "start_ts", "end_ts", "duration_ns", "pcore_idx", "packet_idx", "packet_aggregation_id",
                  "is_host_device_dma", "engine_idx", "read_bytes", "write_bytes", "transfer_bytes"),
    "NkiKernelCallSites": ("execution_index", "nki_call_site_id", "neff_name", "kernel_name", "start_ts", "end_ts"),
    "Warning": None,
    "Error": None,
}
REQUIRED = {
    "Metadata": {"profile_schema_version", "capture_type", "num_physical_cores", "vnc_enabled"},
    "ExecutionInfo": {"execution_index", "neff_name", "execution_start_ts", "execution_end_ts"},
    "Instruction": {"id", "start_ts", "end_ts", "duration_ns", "engine", "engine_instance_idx", "opcode", "pc"},
    "DmaPacket": {"id", "start_ts", "end_ts", "duration_ns", "packet_idx", "is_host_device_dma", "read_bytes", "write_bytes", "transfer_bytes"},
}
WORK_FIELDS = ("raw_flops", "adjusted_flops", "hbm_read_bytes", "hbm_write_bytes")


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def discover(directory):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Neuron profile must be one processed Parquet profile directory")
    files = {}
    for name in TABLES:
        whole = root / f"{name}.parquet"
        shards = sorted(root.glob(name + "_*.parquet"))
        if whole.exists() and shards:
            raise ValueError("Mixed whole and sharded Neuron tables are ambiguous")
        if shards:
            numbers = []
            for path in shards:
                match = re.fullmatch(re.escape(name) + r"_(0|[1-9][0-9]*)\.parquet", path.name)
                if match is None:
                    raise ValueError("Neuron table has an unrecognized shard name")
                numbers.append(int(match[1]))
            if sorted(numbers) != list(range(len(numbers))):
                raise ValueError("Neuron table shards are missing or duplicated")
            paths = sorted(shards, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        else:
            paths = [whole] if whole.exists() else []
        if not paths and name in REQUIRED:
            raise ValueError(f"Neuron profile lacks the complete {name} table")
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise ValueError("Neuron profile tables must be regular local files")
        files[name] = paths
    if sum(map(len, files.values())) > 4096:
        raise ValueError("Neuron profile exceeds the table-file envelope")
    return root, files


def load(directory, *, max_rows=200_000, max_file_bytes=64 * 1024**2, max_decoded_bytes=256 * 1024**2):
    if any(type(v) is not int or v <= 0 for v in (max_rows, max_file_bytes, max_decoded_bytes)):
        raise ValueError("Neuron table envelopes must be positive integers")
    root, files = discover(directory)
    import pyarrow.parquet as pq

    tables, manifest, total_rows, decoded_bytes, file_bytes = {}, {}, 0, 0, 0
    # Inspect every selected shard before decoding any of them.
    pending = []
    for name, paths in files.items():
        if paths:
            tables[name] = []
        for path in paths:
            if path.stat().st_size > max_file_bytes:
                raise ValueError("Neuron table exceeds its compressed-file envelope")
            file_bytes += path.stat().st_size
            if file_bytes > max_decoded_bytes:
                raise ValueError("Neuron profile exceeds its combined file-size envelope")
            before = file_digest(path)
            with pq.ParquetFile(path) as source:
                total_rows += source.metadata.num_rows
                decoded_bytes += source.metadata.serialized_size + sum(source.metadata.row_group(i).total_byte_size for i in range(source.metadata.num_row_groups))
                if total_rows > max_rows or decoded_bytes > max_decoded_bytes:
                    raise ValueError("Full Neuron profile exceeds its row/decoded-memory estimate; no shards omitted")
                names = set(source.schema_arrow.names)
                if len(names) != len(source.schema_arrow.names):
                    raise ValueError("Neuron table has duplicate column identities")
                if not REQUIRED.get(name, set()) <= names:
                    raise ValueError(f"Neuron {name} table omitted required fields")
                selected = sorted(names) if TABLES[name] is None else [k for k in TABLES[name] if k in names]
                manifest[path.name] = {"sha256": before, "bytes": path.stat().st_size, "rows": source.metadata.num_rows,
                                       "columns": selected}
            pending.append((name, path, selected, before))
    for name, path, selected, before in pending:
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(batch_size=4096, columns=selected):
                tables[name].extend(batch.to_pylist())
        if file_digest(path) != before:
            raise ValueError("Neuron profile changed during table decoding")
    _, current = discover(root)
    if {name: [p.name for p in paths] for name, paths in current.items()} != {name: [p.name for p in paths] for name, paths in files.items()}:
        raise ValueError("Neuron table shard inventory changed during decoding")
    # Metadata and event facts, rather than file order, determine interpretation.
    report = analyze(tables)
    report.update(files=manifest, projection="all_rows_of_selected_tables; reviewed_columns_only; no_UI_sampling_or_summary_substitution",
                  resource_envelope={"max_rows": max_rows, "max_file_bytes": max_file_bytes, "max_decoded_bytes": max_decoded_bytes},
                  evaluator_sha256=evaluator_digest())
    report["receipt_sha256"] = fingerprint(report)
    return report


def integer(value, *, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def local_core(value):
    return value is None or (type(value) is int and value == 0)


def interval(row):
    if (not all(integer(row.get(k)) for k in ("start_ts", "end_ts", "duration_ns"))
            or row["end_ts"] < row["start_ts"] or row["duration_ns"] != row["end_ts"] - row["start_ts"]):
        raise ValueError("Neuron event has invalid nanosecond timestamps or duration")
    return row["start_ts"], row["end_ts"]


def indexed(rows, field, label):
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not integer(row.get(field)) for row in rows):
        raise ValueError(f"Neuron {label} requires exact event identities")
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Neuron {label} repeats an execution identity")
    return result


def measured(values):
    if any(value is not None and not integer(value) for value in values):
        raise ValueError("Neuron work must retain exact nonnegative integer values")
    return {"reported_total": sum(value for value in values if value is not None),
            "known_rows": sum(value is not None for value in values), "missing_rows": sum(value is None for value in values),
            "all_rows_reported": all(value is not None for value in values)}


class ExecutionIndex:
    """Find up to two containing executions without scanning every execution."""
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda row: (row["execution_start_ts"], row["execution_index"]))
        self.starts = [row["execution_start_ts"] for row in self.rows]
        self.size = 1 << max(0, (len(self.rows) - 1).bit_length())
        self.ends = [-1] * (2 * self.size)
        for index, row in enumerate(self.rows):
            self.ends[self.size + index] = row["execution_end_ts"]
        for index in range(self.size - 1, 0, -1):
            self.ends[index] = max(self.ends[2 * index], self.ends[2 * index + 1])

    def find(self, start, end):
        stop, matches = bisect_right(self.starts, start), []

        def search(node, left, right):
            if left >= stop or self.ends[node] < end or len(matches) >= 2:
                return
            if right - left == 1:
                matches.append(self.rows[left]["execution_index"])
                return
            middle = (left + right) // 2
            search(2 * node, left, middle)
            search(2 * node + 1, middle, right)

        search(1, 0, self.size)
        return matches[0] if len(matches) == 1 else None


def analyze(tables):
    metadata = tables.get("Metadata")
    if (not isinstance(metadata, list) or len(metadata) != 1 or not isinstance(metadata[0], dict)
            or metadata[0].get("profile_schema_version") != SCHEMA):
        raise ValueError("Neuron work reader requires exactly one Explorer schema 12.0.1 profile")
    metadata = metadata[0]
    if (metadata.get("capture_type") not in {"hardware", "s0", "perf-sim"}
            or type(metadata.get("num_physical_cores")) is not int or metadata["num_physical_cores"] != 1 or metadata.get("vnc_enabled") is not False):
        raise ValueError("Neuron work reader requires an explicit capture kind and LNC=1; no core pooling")
    executions = indexed(tables.get("ExecutionInfo"), "execution_index", "execution table")
    if not executions:
        raise ValueError("Neuron profile contains no observed executions")
    for row in executions.values():
        start, end = row.get("execution_start_ts"), row.get("execution_end_ts")
        if not integer(start) or not integer(end) or end <= start:
            raise ValueError("Old or incomplete execution timelines cannot bind instructions to NEFFs")
        if row.get("neff_name") is not None and not isinstance(row["neff_name"], str):
            raise ValueError("Invalid Neuron executable name")
    instructions = indexed(tables.get("Instruction"), "id", "instruction table")
    packets = indexed(tables.get("DmaPacket"), "id", "DMA packet table")
    if not instructions:
        raise ValueError("Absent device instructions are unknown work, not zero cost")
    calls = {}
    for row in tables.get("NkiKernelCallSites", []):
        if (not isinstance(row, dict) or not integer(row.get("execution_index")) or not integer(row.get("nki_call_site_id"))
                or row["execution_index"] not in executions):
            raise ValueError("NKI call site lacks an observed execution")
        key = row["execution_index"], row["nki_call_site_id"]
        if key in calls:
            raise ValueError("Ambiguous NKI call-site identity")
        if row.get("neff_name") != executions[key[0]].get("neff_name"):
            raise ValueError("NKI call site belongs to another executable")
        calls[key] = row
    all_intervals, engines, joins, unknown_call_sites = [], {}, [], 0
    execution_index = ExecutionIndex(executions.values())

    for row in instructions.values():
        start, end = interval(row)
        if (not isinstance(row.get("engine"), str) or not row["engine"] or not isinstance(row.get("opcode"), str)
                or not row["opcode"] or not integer(row.get("pc")) or not integer(row.get("engine_instance_idx"))
                or not local_core(row.get("pcore_idx"))):
            raise ValueError("Neuron instruction lacks a core/engine/opcode/program-counter identity")
        if end > start:
            all_intervals.append((start, end))
        group = engines.setdefault(row["engine"], [])
        group.append(row)
        declared_execution = row.get("execution_index")
        if declared_execution is not None:
            if not integer(declared_execution) or declared_execution not in executions:
                raise ValueError("Instruction declares an absent or invalid execution")
            owner = executions[declared_execution]
            if not owner["execution_start_ts"] <= start <= end <= owner["execution_end_ts"]:
                raise ValueError("Instruction timestamps contradict its declared execution")
            execution = declared_execution
        else:
            execution = execution_index.find(start, end)
        site = row.get("nki_call_site_id")
        if site is not None and not integer(site):
            raise ValueError("NKI instruction has an invalid call-site ID")
        call = calls.get((execution, site)) if execution is not None and site is not None else None
        unknown_call_sites += int(site is not None and call is None)
        joins.append({"instruction_id": row["id"], "execution_index": execution,
                      "neff_name": executions[execution].get("neff_name") if execution is not None else None,
                      "nki_call_site_id": site, "kernel_name": call.get("kernel_name") if call else None,
                      "origin": "unknown"})
    for row in packets.values():
        start, end = interval(row)
        if (not local_core(row.get("pcore_idx")) or not integer(row.get("packet_idx"))
                or type(row.get("is_host_device_dma")) is not bool
                or any(not integer(row.get(k)) for k in ("read_bytes", "write_bytes", "transfer_bytes"))):
            raise ValueError("Neuron DMA packet lacks exact byte quantities or core identity")
        if end > start:
            all_intervals.append((start, end))
    def busy(rows):
        return interval_union([(row["start_ts"], row["end_ts"]) for row in rows if row["end_ts"] > row["start_ts"]])

    return {"protocol": PROTOCOL, "schema": SCHEMA, "capture_declares_hardware": metadata["capture_type"] == "hardware",
            "metadata": metadata, "execution_count": len(executions), "instruction_count": len(instructions),
            "dma_packet_count": len(packets), "device_busy_ns": interval_union(all_intervals),
            "per_engine": {name: {"instruction_count": len(rows), "busy_ns": busy(rows),
                                  "reported_work": {field: measured([row.get(field) for row in rows]) for field in WORK_FIELDS}}
                           for name, rows in engines.items()},
            "dma_packet_work": {field: measured([row[field] for row in packets.values()]) for field in ("read_bytes", "write_bytes", "transfer_bytes")},
            "joins": joins, "unassigned_instruction_count": sum(row["execution_index"] is None for row in joins),
            "unknown_nki_call_site_count": unknown_call_sites,
            "quality_tables_present": {name: name in tables for name in ("Error", "Warning")},
            "quality_rows": {name: tables.get(name) for name in ("Error", "Warning")},
            "tables_sha256": fingerprint(tables), "tables": tables,
            "scope": "observed_instruction_annotations_and_DMA_packets; no_model_flops_or_summary_rate_inference",
            "hardware_counters_collected": False, "formal_origin_qualified": False, "scored_wall_times_replaced": False}


def validate(report):
    """Recompute every reported value; this does not authenticate device capture."""
    if (not isinstance(report, dict) or report.get("protocol") != PROTOCOL
            or report.get("evaluator_sha256") != evaluator_digest()
            or report.get("receipt_sha256") != fingerprint({k: v for k, v in report.items() if k != "receipt_sha256"})):
        raise ValueError("Neuron work report is missing, stale or changed")
    expected = analyze(report.get("tables", {}))
    if any(report.get(k) != value for k, value in expected.items()):
        raise ValueError("Neuron work totals, missing values or source joins differ from raw tables")
    files, rows, descriptors = report.get("files"), {}, {}
    if not isinstance(files, dict) or not files:
        raise ValueError("Neuron work report has no complete file inventory")
    for name, entry in files.items():
        match = re.fullmatch(r"([A-Za-z]+)(?:_(0|[1-9][0-9]*))?\.parquet", name)
        if (match is None or match[1] not in TABLES or not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                or not integer(entry.get("bytes"), minimum=1) or not integer(entry.get("rows"))
                or not isinstance(entry.get("columns"), list) or any(not isinstance(v, str) for v in entry["columns"])):
            raise ValueError("Neuron work report has an invalid source file descriptor")
        rows[match[1]] = rows.get(match[1], 0) + entry["rows"]
        descriptors.setdefault(match[1], []).append((None if match[2] is None else int(match[2]), entry))
    if set(rows) != set(report["tables"]) or any(count != len(report["tables"][name]) for name, count in rows.items()):
        raise ValueError("Neuron work report omitted rows or table shards")
    for name, entries in descriptors.items():
        parts = [part for part, _ in entries]
        if ((None in parts and len(parts) != 1) or (None not in parts and sorted(parts) != list(range(len(parts))))):
            raise ValueError("Neuron work report has mixed or incomplete source shards")
        offset = 0
        for _, entry in sorted(entries, key=lambda item: -1 if item[0] is None else item[0]):
            columns = set(entry["columns"])
            if (len(columns) != len(entry["columns"]) or not REQUIRED.get(name, set()) <= columns
                    or (TABLES[name] is not None and not columns <= set(TABLES[name]))):
                raise ValueError("Neuron work report has an invalid field projection")
            values = report["tables"][name][offset:offset + entry["rows"]]
            if any(set(row) != columns for row in values):
                raise ValueError("Neuron work report changed fields from its source file projection")
            offset += entry["rows"]
    return report


ENGINE_LABELS = {"TensorMatrix": "tensor", "Vector": "vector", "VectorE": "vector", "Scalar": "scalar",
                 "ScalarE": "scalar", "DMA": "dma", "Dma": "dma", "DMAEngine": "dma"}

UNITS_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}

def timeline(trace, *, timestamp_unit, window, origins=None):
    """Parse neuron-profile view JSON for one declared capture window.

    Source locations alone do not establish authorship. These are diagnostics
    until a trusted NEFF/source association has been established.
    """
    if timestamp_unit not in UNITS_TO_MS or not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError("Neuron timelines require explicit ns/us/ms units and a capture window")
    interval_union([window])
    events = trace.get("instruction")
    if not isinstance(events, list) or not events:
        raise ValueError("No Neuron device instruction data; host/runtime events cannot substitute")
    left, right = window
    rows = []
    for index, event in enumerate(events):
        start, duration = event.get("timestamp"), event.get("duration")
        if type(start) not in (int, float) or not positive(duration):
            raise ValueError("Malformed Neuron instruction timing")
        interval_union([(start, start + duration)])
        if start + duration <= left or start >= right:
            continue
        origin = (origins or {}).get(str(index), "unknown")
        if origin not in {"custom", "library", "unknown"}:
            raise ValueError("Unknown instruction origin")
        rows.append({"start": max(start, left), "end": min(start + duration, right),
                     "engine": ENGINE_LABELS.get(event.get("label"), "unknown"), "origin": origin,
                     "opcode": event.get("opcode"), "nki_source_location": event.get("nki_source_location")})
    if not rows:
        raise ValueError("No device instructions fall inside the declared capture window")
    factor = UNITS_TO_MS[timestamp_unit]

    def union(group):
        return interval_union([(row["start"], row["end"]) for row in group]) * factor

    busy = union(rows)
    other = union([row for row in rows if row["origin"] != "custom"])
    return {"protocol": "neuron_instruction_union_v1", "device_backend": "neuron", "timestamp_unit": timestamp_unit,
            "capture_window": list(window), "capture_span_ms": (right - left) * factor,
            "wall_ms": None, "device_busy_ms": busy, "kernel_count": None, "instruction_count": len(rows),
            "per_engine_busy_ms": {engine: union([row for row in rows if row["engine"] == engine])
                                   for engine in ("tensor", "vector", "scalar", "dma", "unknown")},
            "custom_exclusive_fraction": max(0, busy - other) / busy,
            "unknown_origin_count": sum(row["origin"] == "unknown" for row in rows),
            "origin_map_supplied": origins is not None, "activities": rows,
            "timings_are_diagnostic": True, "hardware_counters_collected": False,
            "scored_wall_times_replaced": False,
            "scope": "instruction_union; not_host_wall_time_or_kernel_count; explicit_capture_binding_required"}

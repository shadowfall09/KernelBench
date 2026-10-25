"""Actual local Parquet I/O and CPU schema controls; no vendor/device execution."""

from __future__ import annotations

import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kb_workflows.common import ROOT, fingerprint
from kb_workflows.hardware import neuron_tables as nt
from scripts.analyze_device_trace import neuron_tables_main as main


def tables():
    # Synthetic schema fixtures. A vendor capture_type string is not attestation.
    def instruction(id, start, end, *, engine="tensor", flops=None, read=0, site=None):
        return {
            "id": id,
            "start_ts": start,
            "end_ts": end,
            "duration_ns": end - start,
            "engine": engine,
            "engine_instance_idx": 0,
            "opcode": "MATMUL" if engine == "tensor" else "ADD",
            "pc": id,
            "pcore_idx": 0,
            "raw_flops": flops,
            "adjusted_flops": None if flops is None else flops * 4,
            "hbm_read_bytes": read,
            "hbm_write_bytes": read,
            "nki_call_site_id": site,
        }

    def packet(id, start, end, *, host=False):
        return {
            "id": id,
            "start_ts": start,
            "end_ts": end,
            "duration_ns": end - start,
            "packet_idx": 0,
            "is_host_device_dma": host,
            "read_bytes": 32,
            "write_bytes": 32,
            "transfer_bytes": 32,
        }

    return {
        "Metadata": [
            {
                "profile_schema_version": nt.SCHEMA,
                "capture_type": "hardware",
                "num_physical_cores": 1,
                "vnc_enabled": False,
                "explorer_version": "unscored schema fixture",
            }
        ],
        "ExecutionInfo": [
            {"execution_index": 0, "execution_start_ts": 0, "execution_end_ts": 100, "neff_name": "a.neff"},
            {"execution_index": 1, "execution_start_ts": 100, "execution_end_ts": 200, "neff_name": "b.neff"},
        ],
        "Instruction": [
            instruction(2**53 + 3, 10, 30, flops=2**53 + 3, read=64, site=7),
            instruction(2, 20, 50, engine="vector"),
            instruction(3, 110, 160, flops=120, read=None, site=7),
        ],
        "DmaPacket": [packet(1, 25, 55), packet(2, 140, 180, host=True)],
        "NkiKernelCallSites": [
            {"execution_index": i, "nki_call_site_id": 7, "neff_name": f"{name}.neff", "kernel_name": "same_label"}
            for i, name in enumerate(("a", "b"))
        ],
        "Error": [],
        "Warning": [],
    }


def write(directory, values=None, *, sharded=False):
    directory.mkdir()
    values = values or tables()
    for name, rows in values.items():
        table = pa.Table.from_pylist(rows)
        if name == "Instruction" and sharded:
            pq.write_table(table.slice(0, 2), directory / "Instruction_0.parquet")
            pq.write_table(table.slice(2), directory / "Instruction_1.parquet")
        else:
            pq.write_table(table, directory / f"{name}.parquet")
    # Static summaries cannot supply missing observed work.
    pq.write_table(pa.table({"hardware_flops": [1], "model_flops": [2**62]}), directory / "Summary.parquet")
    return directory


@pytest.mark.parametrize("sharded", [False, True])
def test_real_parquet_rows_preserve_large_integers_overlap_and_execution_scoped_nki_ids(tmp_path, sharded):
    directory = write(tmp_path / "profile", sharded=sharded)
    report = nt.load(directory)
    assert report["instruction_count"] == 3 and report["dma_packet_count"] == 2
    assert report["device_busy_ns"] == 115
    assert report["per_engine"]["tensor"]["busy_ns"] == 70
    measured = report["per_engine"]["tensor"]["reported_work"]
    assert measured["raw_flops"]["reported_total"] == 2**53 + 3 + 120
    assert measured["hbm_read_bytes"] == {
        "reported_total": 64,
        "known_rows": 1,
        "missing_rows": 1,
        "all_rows_reported": False,
    }
    assert report["per_engine"]["vector"]["reported_work"]["raw_flops"]["missing_rows"] == 1
    assert report["joins"][0]["neff_name"] == "a.neff" and report["joins"][2]["neff_name"] == "b.neff"
    assert report["unknown_nki_call_site_count"] == 0
    assert all(row["origin"] == "unknown" for row in report["joins"])
    assert (
        report["capture_declares_hardware"]
        and not report["hardware_counters_collected"]
        and not report["formal_origin_qualified"]
    )
    assert "Summary.parquet" not in report["files"] and nt.validate(json.loads(json.dumps(report))) == report


@pytest.mark.parametrize("kind", ["s0", "perf-sim"])
def test_simulator_profile_annotations_never_become_hardware_measurements(tmp_path, kind):
    data = tables()
    data["Metadata"][0]["capture_type"] = kind
    report = nt.load(write(tmp_path / "simulation", data))
    assert not report["capture_declares_hardware"] and not report["hardware_counters_collected"]


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "missing_kind",
        "multiple_cores",
        "bool_cores",
        "missing_execution",
        "old_timeline",
        "duplicate_instruction",
        "duration",
        "float_work",
        "negative_work",
        "core",
        "dtype_id",
        "duplicate_site",
        "wrong_neff",
        "duplicate_packet",
        "missing_bytes",
        "packet_core",
        "bool_core",
        "float_packet_core",
    ],
)
def test_incomplete_or_ambiguous_vendor_facts_cannot_be_repaired_by_summaries(damage):
    data = tables()
    if damage == "schema":
        data["Metadata"][0]["profile_schema_version"] = "7.4.0"
    elif damage == "missing_kind":
        data["Metadata"][0]["capture_type"] = None
    elif damage == "multiple_cores":
        data["Metadata"][0]["num_physical_cores"] = 2
    elif damage == "bool_cores":
        data["Metadata"][0]["num_physical_cores"] = True
    elif damage == "missing_execution":
        data["ExecutionInfo"] = []
    elif damage == "old_timeline":
        data["ExecutionInfo"][0]["execution_start_ts"] = None
    elif damage == "duplicate_instruction":
        data["Instruction"].append(data["Instruction"][0])
    elif damage == "duration":
        data["Instruction"][0]["duration_ns"] = 1
    elif damage == "float_work":
        data["Instruction"][0]["raw_flops"] = 1.5
    elif damage == "negative_work":
        data["Instruction"][0]["raw_flops"] = -1
    elif damage == "core":
        data["Instruction"][0]["pcore_idx"] = 1
    elif damage == "dtype_id":
        data["Instruction"][0]["id"] = float(2**53 + 3)
    elif damage == "duplicate_site":
        data["NkiKernelCallSites"].append(data["NkiKernelCallSites"][0])
    elif damage == "wrong_neff":
        data["NkiKernelCallSites"][0]["neff_name"] = "wrong.neff"
    elif damage == "duplicate_packet":
        data["DmaPacket"].append(data["DmaPacket"][0])
    elif damage == "missing_bytes":
        data["DmaPacket"][0]["transfer_bytes"] = None
    elif damage == "packet_core":
        data["DmaPacket"][0]["pcore_idx"] = 1
    elif damage == "bool_core":
        data["Instruction"][0]["pcore_idx"] = False
    elif damage == "float_packet_core":
        data["DmaPacket"][0]["pcore_idx"] = 0.0
    with pytest.raises(ValueError):
        nt.analyze(data)


def test_overlapping_execution_ranges_and_absent_quality_tables_remain_explicitly_unknown():
    data = tables()
    data["ExecutionInfo"][1]["execution_start_ts"] = 15
    del data["Error"]
    del data["Warning"]
    report = nt.analyze(data)
    assert report["unassigned_instruction_count"] == 1 and report["unknown_nki_call_site_count"] == 0
    assert report["quality_tables_present"] == {"Error": False, "Warning": False}
    assert report["quality_rows"] == {"Error": None, "Warning": None}


def test_execution_interval_index_handles_nested_ranges_and_zero_duration_boundaries_exactly():
    rows = [
        {"execution_index": 0, "execution_start_ts": 0, "execution_end_ts": 1000},
        {"execution_index": 1, "execution_start_ts": 10, "execution_end_ts": 20},
        {"execution_index": 2, "execution_start_ts": 1000, "execution_end_ts": 2000},
    ]
    index = nt.ExecutionIndex(rows)
    assert index.find(300, 400) == 0 and index.find(10, 11) is None
    assert index.find(1000, 1000) is None and index.find(1000, 1001) == 2
    assert index.find(999, 1001) is None


@pytest.mark.parametrize("damage", ["missing_shard", "mixed", "symlink", "missing_table", "missing_column"])
def test_file_inventory_cannot_drop_or_mix_selected_shards(tmp_path, damage):
    directory = write(tmp_path / "profile", sharded=True)
    if damage == "missing_shard":
        (directory / "Instruction_0.parquet").unlink()
    elif damage == "mixed":
        pq.write_table(pa.Table.from_pylist(tables()["Instruction"]), directory / "Instruction.parquet")
    elif damage == "symlink":
        saved = tmp_path / "outside.parquet"
        (directory / "Instruction_0.parquet").rename(saved)
        (directory / "Instruction_0.parquet").symlink_to(saved)
    elif damage == "missing_table":
        (directory / "DmaPacket.parquet").unlink()
    elif damage == "missing_column":
        pq.write_table(pa.table({"id": [1]}), directory / "Instruction_0.parquet")
    with pytest.raises(ValueError):
        nt.load(directory)


@pytest.mark.parametrize("limits", [{"max_rows": 1}, {"max_file_bytes": 1}, {"max_decoded_bytes": 1}])
def test_full_profile_resource_envelopes_reject_before_any_rows_are_decoded(tmp_path, monkeypatch, limits):
    directory = write(tmp_path / "profile")
    monkeypatch.setattr(
        pq.ParquetFile, "iter_batches", lambda *args, **kwargs: pytest.fail("decoded past admission failure")
    )
    with pytest.raises(ValueError, match="envelope|estimate"):
        nt.load(directory, **limits)


@pytest.mark.parametrize("damage", ["total", "join", "quality", "hardware", "file_rows", "raw_table", "file_columns", "file_shard"])
def test_resealed_reports_cannot_change_work_missingness_or_source_joins(tmp_path, damage):
    report = nt.load(write(tmp_path / "profile"))
    if damage == "total":
        report["per_engine"]["tensor"]["reported_work"]["raw_flops"]["reported_total"] += 1
    elif damage == "join":
        report["joins"][0]["neff_name"] = "b.neff"
    elif damage == "quality":
        report["quality_tables_present"]["Error"] = False
    elif damage == "hardware":
        report["hardware_counters_collected"] = True
    elif damage == "file_rows":
        report["files"]["Instruction.parquet"]["rows"] -= 1
    elif damage == "raw_table":
        report["tables"]["Instruction"][0]["raw_flops"] = None
    elif damage == "file_columns":
        report["files"]["Instruction.parquet"]["columns"] = ["id"]
    elif damage == "file_shard":
        report["files"]["Instruction_2.parquet"] = report["files"].pop("Instruction.parquet")
    report["receipt_sha256"] = fingerprint({k: v for k, v in report.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError):
        nt.validate(report)


def test_offline_command_and_imports_do_not_start_vendor_tools_or_import_device_sdks(tmp_path, monkeypatch):
    directory = write(tmp_path / "profile")
    output = tmp_path / "work.json"
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("offline analysis launched a tool"))
    assert main([str(directory), "--output", str(output)]) == 0
    nt.validate(json.loads(output.read_text()))
    assert main([str(directory), "--output", str(output)]) == 2


def test_parser_module_import_remains_sdk_and_arrow_light():
    code = (
        "import sys; import kb_workflows.hardware.neuron_tables, scripts.analyze_device_trace; "
        "assert not any(n in sys.modules for n in ('torch','pyarrow','triton','torch_neuronx'))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)

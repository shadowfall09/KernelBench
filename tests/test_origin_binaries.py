"""Installed-binary provenance with synthetic ELF files; no SDK/tool execution."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from kb_workflows import origin_binaries as binaries
from kb_workflows.common import fingerprint


def cuda_elf(payload=b"synthetic cubin"):
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = (190).to_bytes(2, "little")
    return bytes(header) + payload


def distribution(tmp_path, *, name="torch", content=b"frozen host DSO", relative="torch/lib/test.so.1"):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
    output = io.StringIO()
    csv.writer(output).writerow([relative, "sha256=" + digest, len(content)])
    record = output.getvalue()
    dist = NS(metadata={"Name": name}, version="1.0", read_text=lambda key: record if key == "RECORD" else None,
              locate_file=lambda relative: tmp_path / relative)
    return dist, path


def extractor(tmp_path, *, output=None, timeout=False, exit_code=0):
    path = tmp_path / "fake-cuobjdump"
    path.write_text("#!/bin/sh\nexit 99\n")
    path.chmod(0o700)
    calls = []

    def run(command, directory, budget):
        assert command[1:3] == ["--extract-elf", "all"]
        assert 0 < budget <= 30
        calls.append(command)
        if output is not None:
            (directory / "arbitrary.output.name").write_bytes(output)
        log = b"extracting bytes\x00\xff\n"
        return {"exit_code": exit_code, "timed_out": timeout, "log_sha256": hashlib.sha256(log).hexdigest(),
                "log_base64": base64.b64encode(log).decode()}

    return str(path), run, calls


def catalog(tmp_path, *, output=None, **options):
    dist, path = distribution(tmp_path)
    tool, run, calls = extractor(tmp_path, output=output, **options)
    obj = binaries.Catalog(frozen=binaries.manifest(distributions=[dist]), command=tool, runner=run)
    return obj, path, calls


def test_freeze_uses_record_sha256_not_a_path_or_declared_package_prefix(tmp_path):
    dist, path = distribution(tmp_path)
    frozen = binaries.manifest(distributions=[dist])
    source = frozen["sources"][0]
    assert source["path"] == str(path) and source["origin"] == "framework"
    assert source["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen["manifest_sha256"] == fingerprint({"sources": frozen["sources"]})
    outsider, _ = distribution(tmp_path / "other", name="candidate-cublas")
    assert binaries.manifest(distributions=[outsider])["sources"] == []


@pytest.mark.parametrize("name,owner", [("torch", "framework"), ("nvidia-cublas", "library"),
                                      ("nvidia-cudnn-cu13", "library"), ("nvidia_cufft_cu12", "library")])
def test_known_distribution_names_use_explicit_roles(tmp_path, name, owner):
    dist, _ = distribution(tmp_path, name=name)
    assert binaries.manifest(distributions=[dist])["sources"][0]["origin"] == owner


def test_manifest_rejects_unhashed_missing_symlink_and_outside_files(tmp_path):
    dist, path = distribution(tmp_path)
    raw = dist.read_text("RECORD")
    dist.read_text = lambda key: raw.replace("sha256=", "md5=")
    assert binaries.manifest(distributions=[dist])["sources"] == []
    dist.read_text = lambda key: raw
    path.unlink()
    assert binaries.manifest(distributions=[dist])["sources"] == []
    other = tmp_path / "elsewhere.so"
    other.write_bytes(b"frozen host DSO")
    path.symlink_to(other)
    assert binaries.manifest(distributions=[dist])["sources"] == []


def test_only_loaded_record_bound_dsos_are_extracted(tmp_path):
    cubin = cuda_elf()
    obj, path, calls = catalog(tmp_path, output=cubin)
    empty = obj.collect(paths=[])
    assert not calls and empty["sources"] == []
    document = obj.collect(paths=[str(path)])
    assert len(calls) == 1 and document["sources"][0]["status"] == "measured"
    assert document["cubins"][0]["binary_sha256"] == hashlib.sha256(cubin).hexdigest()
    assert binaries.binary_owners(document) == {}  # Injected runner stays simulated.
    assert binaries.binary_owners(document, allow_simulated=True) == {hashlib.sha256(cubin).hexdigest(): "framework"}
    assert json.loads(json.dumps(document)) == document


def test_modified_installed_bytes_cannot_reuse_record_identity(tmp_path):
    obj, path, calls = catalog(tmp_path, output=cuda_elf())
    path.write_bytes(b"candidate substitution")
    document = obj.collect(paths=[str(path)])
    assert not calls and document["sources"][0]["status"] == "not_measured"
    assert "no longer matches" in document["errors"][0]["reason"]
    assert not binaries.binary_owners(document, allow_simulated=True)


@pytest.mark.parametrize("output", [b"plain text", b"\x7fELF\x02\x01", cuda_elf()[:18] + b"\x3e\x00" + cuda_elf()[20:]])
def test_non_cuda_or_partial_extractor_output_never_establishes_ownership(tmp_path, output):
    obj, path, _ = catalog(tmp_path, output=output)
    document = obj.collect(paths=[str(path)])
    assert document["sources"][0]["status"] == "not_measured" and document["cubins"] == []


@pytest.mark.parametrize("options", [{"timeout": True}, {"exit_code": 1}])
def test_failed_or_timed_out_tool_has_no_completed_binary_proofs(tmp_path, options):
    obj, path, _ = catalog(tmp_path, output=cuda_elf(), **options)
    document = obj.collect(paths=[str(path)])
    assert document["cubins"] == [] and document["sources"][0]["status"] == "not_measured"


def test_source_changed_during_extraction_discards_all_pending_cubins(tmp_path):
    obj, path, _ = catalog(tmp_path, output=cuda_elf())
    original = obj.runner

    def tamper(*args):
        response = original(*args)
        path.write_bytes(b"changed during extraction")
        return response

    obj.runner = tamper
    document = obj.collect(paths=[str(path)])
    assert document["cubins"] == [] and "changed during" in document["errors"][0]["reason"]


def test_tool_identity_is_frozen_before_candidate_changes_it(tmp_path):
    obj, path, calls = catalog(tmp_path, output=cuda_elf())
    Path(obj.command).write_text("changed tool")
    document = obj.collect(paths=[str(path)])
    assert not calls and document["cubins"] == []


def test_missing_tool_remains_unknown_and_does_not_infer_from_loaded_names(tmp_path):
    dist, path = distribution(tmp_path)
    obj = binaries.Catalog(frozen=binaries.manifest(distributions=[dist]), command=str(tmp_path / "missing"))
    document = obj.collect(paths=[str(path)])
    assert document["tool"] is None and document["errors"] and binaries.binary_owners(document) == {}


def test_extractor_symlinks_and_fifo_outputs_are_rejected_without_blocking(tmp_path):
    import os

    for kind in ("symlink", "fifo"):
        root = tmp_path / kind
        root.mkdir()
        obj, path, _ = catalog(root)
        original = obj.runner

        def special(command, directory, budget):
            response = original(command, directory, budget)
            output = directory / "forged.cubin"
            if kind == "symlink":
                output.symlink_to(path)
            else:
                os.mkfifo(output)
            return response

        obj.runner = special
        assert obj.collect(paths=[str(path)])["cubins"] == []


def test_tampering_manifest_receipt_raw_log_or_source_status_is_unknown(tmp_path):
    obj, path, _ = catalog(tmp_path, output=cuda_elf())
    document = obj.collect(paths=[str(path)])
    for mutate in (
        lambda d: d["manifest"]["sources"][0].update(origin="custom"),
        lambda d: d["sources"][0].update(status="not_measured"),
        lambda d: d["sources"][0]["extraction"].update(log_base64="Zm9yZ2Vk"),
        lambda d: d.update(cubins=[{"source": 99, "binary_sha256": "1" * 64, "binary_bytes": 1}]),
    ):
        changed = copy.deepcopy(document)
        mutate(changed)
        changed["receipt_sha256"] = fingerprint({k: v for k, v in changed.items() if k != "receipt_sha256"})
        assert binaries.binary_owners(changed, allow_simulated=True) == {}


def test_conflicting_approved_binary_owners_remain_explicitly_unknown(tmp_path):
    obj, path, _ = catalog(tmp_path, output=cuda_elf())
    document = obj.collect(paths=[str(path)])
    source = {**document["manifest"]["sources"][0], "id": 2, "distribution": "nvidia-cublas", "origin": "library"}
    document["manifest"]["sources"].append(source)
    document["manifest"]["manifest_sha256"] = fingerprint({"sources": document["manifest"]["sources"]})
    document["sources"].append({**document["sources"][0], "source": 2})
    document["cubins"].append({**document["cubins"][0], "source": 2})
    document["receipt_sha256"] = fingerprint({k: v for k, v in document.items() if k != "receipt_sha256"})
    assert binaries.binary_owners(document, allow_simulated=True) == {hashlib.sha256(cuda_elf()).hexdigest(): "unknown"}


def test_loaded_maps_parser_preserves_space_names_and_excludes_deleted_dsos(tmp_path):
    path = tmp_path / "maps"
    path.write_text("1-2 r-xp 00000000 00:00 1 /wheel/with space/lib.so.13\n"
                    "3-4 rw-p 00000000 00:00 2 /wheel/deleted.so (deleted)\n"
                    "5-6 r-xp 00000000 00:00 3 /wheel/not-a-dso\n")
    assert binaries.loaded_dsos(path) == {"/wheel/with space/lib.so.13"}


@pytest.mark.parametrize("document", [None, [], "custom", {}, {"protocol": binaries.PROTOCOL, "manifest": None},
                                     {"protocol": binaries.PROTOCOL, "sources": [None]}])
def test_malformed_library_documents_never_establish_an_owner(document):
    assert binaries.binary_owners(document) == {}

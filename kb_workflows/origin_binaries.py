"""Verifier-owned wheel binary manifests and bounded offline cubin extraction.

Freeze installed-package RECORD identities before candidate import. Only loaded
DSOs whose bytes match that frozen identity can supply library/framework cubins.
This is installed-image provenance, not a signature or a Python sandbox.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from .common import fingerprint

PROTOCOL = "kernelbench_pro_wheel_cubins_v1"
MAX_SOURCES = 1024
MAX_CUBINS = 20_000
MAX_CUBIN_BYTES = 128 * 1024**2
MAX_LOG_BYTES = 4 * 1024**2
MAX_DSO_BYTES = 32 * 1024**3

# CUDA 13 wheels use both bare and suffixed distribution names. No candidate-
# supplied package name, path prefix or origin label extends this allowlist.
PACKAGES = {"torch": "framework"}
for _name in ("cublas", "cudnn", "cufft", "cusolver", "cusparse", "curand", "nccl", "nvjitlink"):
    for _suffix in ("", "-cu12", "-cu13"):
        PACKAGES[f"nvidia-{_name}{_suffix}"] = "library"


def file_digest(path, *, limit=MAX_DSO_BYTES):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("Binary must be a bounded nonempty regular file")
        digest = hashlib.sha256()
        count = 0
        while chunk := handle.read(min(1024**2, limit + 1 - count)):
            count += len(chunk)
            if count > limit:
                raise ValueError("Binary grew beyond its bound")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        if count != info.st_size or (info.st_size, info.st_mtime_ns, info.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("Binary changed while hashing")
    return digest.hexdigest(), count


def loaded_dsos(maps_path=Path("/proc/self/maps")):
    paths = set()
    for line in maps_path.read_text().splitlines():
        pieces = line.split(maxsplit=5)
        if len(pieces) == 6 and pieces[5].startswith("/") and not pieces[5].endswith(" (deleted)"):
            path = Path(pieces[5])
            if re.search(r"\.so(?:\.\d+)*$", path.name):
                paths.add(str(path))
    return paths


def manifest(*, distributions=None):
    """Read only distribution metadata; do not import Torch or load a DSO."""
    supplied = distributions is not None
    distributions = distributions or []
    if not supplied:
        for name in PACKAGES:
            try:
                distributions.append(importlib.metadata.distribution(name))
            except importlib.metadata.PackageNotFoundError:
                pass
    rows, seen = [], set()
    for dist in distributions:
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name not in PACKAGES:
            continue
        record = dist.read_text("RECORD")
        if not isinstance(record, str):
            continue
        root = Path(dist.locate_file("")).resolve()
        record_sha = hashlib.sha256(record.encode()).hexdigest()
        for relative, digest, size in csv.reader(io.StringIO(record)):
            if not re.search(r"\.so(?:\.\d+)*$", relative) or not digest.startswith("sha256="):
                continue
            candidate = Path(dist.locate_file(relative))
            try:
                resolved = candidate.resolve(strict=True)
                expected = base64.b64decode(digest.removeprefix("sha256=") + "=", altchars=b"-_", validate=True).hex()
                length = int(size)
                if (candidate.is_symlink() or not resolved.is_relative_to(root) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                        or not 0 < length <= MAX_DSO_BYTES or str(resolved) in seen):
                    continue
            except (OSError, ValueError):
                continue
            rows.append({"id": len(rows) + 1, "distribution": name, "version": dist.version,
                "origin": PACKAGES[name], "record_sha256": record_sha, "relative_path": relative,
                "path": str(resolved), "file_sha256": expected, "file_bytes": length})
            seen.add(str(resolved))
            if len(rows) > MAX_SOURCES:
                raise ValueError("Excessive installed binary manifest")
    return {"protocol": PROTOCOL, "sources": rows, "manifest_sha256": fingerprint({"sources": rows})}


def _run_tool(command, directory, timeout):
    from .transports import terminate

    log = directory / "cuobjdump.log"
    with log.open("wb") as output:
        process = subprocess.Popen(command, cwd=directory, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "timed_out": True, "log_sha256": None}
        finally:
            terminate(process)
    if log.stat().st_size > MAX_LOG_BYTES:
        return {"exit_code": process.returncode, "timed_out": False, "log_sha256": None}
    text = log.read_bytes()
    return {"exit_code": process.returncode, "timed_out": False,
            "log_sha256": hashlib.sha256(text).hexdigest(), "log_base64": base64.b64encode(text).decode()}


class Catalog:
    def __init__(self, *, frozen=None, command="cuobjdump", runner=None):
        self.manifest = frozen if frozen is not None else manifest()
        if self.manifest.get("protocol") != PROTOCOL or self.manifest.get("manifest_sha256") != fingerprint({"sources": self.manifest.get("sources")}):
            raise ValueError("Unbound wheel manifest")
        resolved = shutil.which(command)
        self.command = str(Path(resolved).resolve()) if resolved else None
        self.runner = runner or _run_tool
        self.simulated = runner is not None
        self.tool = None
        if self.command:
            digest, size = file_digest(self.command)
            self.tool = {"path": self.command, "sha256": digest, "bytes": size}

    def collect(self, *, paths=None, timeout=60):
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise ValueError("Binary extraction budget must be in (0, 300] seconds")
        selected = set(paths) if paths is not None else loaded_dsos()
        result = {"protocol": PROTOCOL, "manifest": self.manifest, "tool": self.tool,
                  "sources": [], "cubins": [], "errors": [], "simulated": self.simulated}
        if self.tool is None:
            result["errors"].append("cuobjdump is unavailable; paths and names cannot substitute")
            return result
        deadline = time.monotonic() + timeout
        for source in self.manifest["sources"]:
            if source["path"] not in selected:
                continue
            record = {"source": source["id"], "status": "not_measured", "extraction": None}
            result["sources"].append(record)
            try:
                expected = (source["file_sha256"], source["file_bytes"])
                if file_digest(source["path"]) != expected or file_digest(self.command)[0] != self.tool["sha256"]:
                    raise ValueError("Installed DSO or tool no longer matches the frozen identity")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("Binary extraction deadline exhausted")
                with tempfile.TemporaryDirectory(prefix="kb-origin-cubins-") as temporary:
                    directory = Path(temporary)
                    response = self.runner([self.command, "--extract-elf", "all", source["path"]], directory, min(remaining, 30))
                    record["extraction"] = response
                    if (response.get("exit_code") != 0 or response.get("timed_out") is not False
                            or not re.fullmatch(r"[0-9a-f]{64}", str(response.get("log_sha256", "")))):
                        raise ValueError("CUBIN extraction did not complete with a retained bounded log")
                    pending = []
                    for path in sorted(directory.iterdir()):
                        if path.name == "cuobjdump.log":
                            continue
                        digest, size = file_digest(path, limit=MAX_CUBIN_BYTES)
                        with path.open("rb") as handle:
                            header = handle.read(20)
                        # CUDA cubins use ELF e_machine=190 (EM_CUDA). Host ELF,
                        # arbitrary text, output names and partial files cannot
                        # establish a device-binary identity.
                        if (len(header) < 20 or header[:4] != b"\x7fELF" or header[4] not in {1, 2}
                                or header[5] not in {1, 2}
                                or int.from_bytes(header[18:20], "little" if header[5] == 1 else "big") != 190):
                            raise ValueError("Extractor output is not a CUDA ELF binary")
                        pending.append({"source": source["id"], "binary_sha256": digest, "binary_bytes": size})
                        if len(result["cubins"]) + len(pending) > MAX_CUBINS:
                            raise ValueError("Extracted CUBIN count exceeded its bound")
                    if file_digest(source["path"]) != expected or file_digest(self.command)[0] != self.tool["sha256"]:
                        raise ValueError("DSO or tool changed during binary extraction")
                    result["cubins"].extend(pending)
                    record["status"] = "measured"
            except (OSError, ValueError, RuntimeError) as exc:
                record["reason"] = str(exc)
                result["errors"].append({"source": source["id"], "reason": str(exc)})
        result["receipt_sha256"] = fingerprint(result)
        return result


def _binary_owners(document, *, allow_simulated=False):
    """Recompute trusted ownership only from completed source-specific receipts."""
    if not isinstance(document, dict) or document.get("protocol") != PROTOCOL:
        return {}
    receipt = {key: value for key, value in document.items() if key != "receipt_sha256"}
    frozen = document.get("manifest", {})
    if (document.get("receipt_sha256") != fingerprint(receipt)
            or frozen.get("protocol") != PROTOCOL
            or frozen.get("manifest_sha256") != fingerprint({"sources": frozen.get("sources")})
            or (document.get("simulated") is not False and not allow_simulated)
            or not isinstance(document.get("tool"), dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(document["tool"].get("sha256")))):
        return {}
    if (not isinstance(frozen.get("sources"), list) or len(frozen["sources"]) > MAX_SOURCES
            or not isinstance(document.get("cubins"), list) or len(document["cubins"]) > MAX_CUBINS
            or not isinstance(document.get("sources"), list) or len(document["sources"]) > MAX_SOURCES):
        return {}
    sources = {}
    for row in frozen.get("sources", []):
        if (not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0 or row["id"] in sources
                or row.get("distribution") not in PACKAGES or row.get("origin") != PACKAGES[row["distribution"]]
                or not row.get("version")
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get(key))) for key in ("record_sha256", "file_sha256"))):
            return {}
        sources[row["id"]] = row
    complete, seen = set(), set()
    for row in document.get("sources", []):
        if (row.get("source") in seen or row.get("source") not in sources):
            return {}
        seen.add(row["source"])
        extraction = row.get("extraction", {}) or {}
        log = base64.b64decode(extraction.get("log_base64", ""), validate=True)
        if (row.get("status") == "measured" and extraction.get("exit_code") == 0
                and extraction.get("timed_out") is False
                and len(log) <= MAX_LOG_BYTES and hashlib.sha256(log).hexdigest() == extraction.get("log_sha256")):
            complete.add(row["source"])
    owners = {}
    for row in document.get("cubins", []):
        if (row.get("source") not in complete or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("binary_sha256")))
                or type(row.get("binary_bytes")) is not int or not 0 < row["binary_bytes"] <= MAX_CUBIN_BYTES):
            return {}
        owners.setdefault(row["binary_sha256"], set()).add(sources[row["source"]]["origin"])
    # Do not omit a conflict: omission could otherwise fall back to custom
    # compiler ownership for copied bytes with ambiguous installed provenance.
    return {digest: next(iter(kinds)) if len(kinds) == 1 else "unknown" for digest, kinds in owners.items()}


def binary_owners(document, *, allow_simulated=False):
    """Malformed/incomplete receipts cannot establish an installed binary owner."""
    try:
        return _binary_owners(document, allow_simulated=allow_simulated)
    except (ValueError, TypeError, KeyError, AttributeError):
        return {}

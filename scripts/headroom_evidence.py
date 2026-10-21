"""Normalize a supplied research archive without executing any archived code.

Run from the checkout: python -m scripts.headroom_evidence --archive FILE.zip
The default output is research/headroom/evidence; this never changes the suite registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

PREFIX = "headroom-bench/"
TABLES = ("ledger_compact.csv", "l1_value_table.csv", "l2_l3_value_table.csv", "sakana_l1_sol.csv")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(archive, output):
    """Return import provenance; retain audit judgments as judgments, never GPU verdicts."""
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"archive": archive.name, "archive_sha256": digest(archive.read_bytes()),
                  "candidate_gpu_runs": False, "raw_candidate_sources_included": False, "sources": {},
                  "omitted": ["historical drafts", "internal conversation notes", "duplicate checklists",
                              "unexecuted attack scripts and hardening prototype", "machine-specific scripts"],
                  "historical_analysis": "Archive tables are not recomputed for the current checkout"}
    with ZipFile(archive) as bundle:
        def read(name):
            raw = bundle.read(PREFIX + name)
            provenance["sources"][name] = {"sha256": digest(raw), "bytes": len(raw)}
            return raw

        for name in ("PAPER.md", "MOTIVATION_ZH.md", "MOTIVATION_EN.md", "INTERNAL_NOTES.md",
                     "audit/REPORT.md", "audit/eval_hardening/strict_check.py"):
            read(name)  # Record the source; rewrite the narrative instead of copying drafts.
        rows = list(csv.DictReader(io.StringIO(read("evidence/kb100.csv").decode("utf-8-sig"))))
        counts = Counter(row["role"] for row in rows)
        if len({(row["level"], row["pid"]) for row in rows}) != 250 or counts != {"scored": 105, "probe": 5, "extended": 140}:
            raise ValueError("Unexpected KernelBench-105 selection")
        with (output / "selection.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (int(row["level"]), int(row["pid"]))))
        findings = json.loads(read("audit/data/findings.json"))
        stats = json.loads(read("audit/data/statistics.json"))
        verdicts = Counter(row["severity"] for row in findings["implementations"])
        if len(findings["implementations"]) != stats["implementations"] or dict(verdicts) != stats["by_severity"]:
            raise ValueError("Audit severity totals do not agree")
        with (output / "audit_implementations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["key", "task", "precision", "model", "displayed_speedup",
                                                        "source_review_verdict", "flags", "notes", "basis",
                                                        "archived_source_path", "raw_source_available"], lineterminator="\n")
            writer.writeheader()
            for row in sorted(findings["implementations"], key=lambda item: item["key"]):
                writer.writerow({"key": row["key"], "task": row["task"], "precision": row["precision"],
                                 "model": row["model"], "displayed_speedup": row["speedup"],
                                 "source_review_verdict": row["severity"],
                                 "flags": ";".join(flag["flag"] for flag in row["flags"]),
                                 "notes": " | ".join(flag["note"] for flag in row["flags"]),
                                 "basis": " | ".join(flag["basis"] for flag in row["flags"]),
                                 "archived_source_path": row["source_file"], "raw_source_available": False})
        (output / "audit_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
        for name in TABLES:
            raw = read("evidence/" + name)
            (output / name).write_text(raw.decode("utf-8-sig").replace("\r\n", "\n"))
    provenance["outputs"] = {path.name: digest(path.read_bytes()) for path in sorted(output.iterdir())
                              if path.is_file() and path.name != "source_manifest.json"}
    (output / "source_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("research/headroom/evidence"))
    args = parser.parse_args()
    provenance = normalize(args.archive, args.output)
    print(json.dumps({"archive_sha256": provenance["archive_sha256"], "outputs": list(provenance["outputs"])}, indent=2))


if __name__ == "__main__":
    main()

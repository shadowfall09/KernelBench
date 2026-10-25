# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render current native scores, grouped by compatible experiment settings."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from kb_workflows.analysis.experiment import compare_runs
from kb_workflows.analysis.paper_analysis import analyze_paper
from kb_workflows.common import read_json
from kb_workflows.config import RunConfig


def escaped(value):
    return html.escape(str(value), quote=True)


def render(directories):
    reports = {}
    for directory in map(Path, directories):
        directory = directory.resolve()
        settings = read_json(directory / "run_config.json")
        if not isinstance(settings, dict):
            raise ValueError(f"Missing run configuration: {directory}")
        config = RunConfig(**settings)
        config.runs_dir, config.run_name = str(directory.parent), directory.name
        config.validate()
        # Recompute every gate from saved evidence. Rendering never runs a kernel.
        reports[directory] = analyze_paper(config)
    comparison = compare_runs(reports)
    sections = []
    for index, group in enumerate(comparison["groups"], 1):
        contract = group["contract"]
        members = []
        for run in group["runs"]:
            directory = Path(run["directory"])
            report = reports[directory]
            details = "".join(
                "<tr>" + "".join(f"<td>{escaped(row.get(key, ''))}</td>" for key in
                                ("problem_id", "sample_id", "layer", "status", "score")) + "</tr>"
                for row in report["details"]
            )
            eligibility = "eligible" if report["leaderboard_eligible"] else "diagnostic"
            members.append(
                f"<article><h3>{escaped(directory)}</h3>"
                f"<p>Model: {escaped(run['declared_model'])}; score: {escaped(report['score_macro'])}; "
                f"status: {eligibility}; declared samples per problem: {report['declared_samples']}</p>"
                f"<p>Aggregation: {escaped(report['aggregation'])}</p>"
                "<details><summary>Every declared sample</summary><table>"
                "<thead><tr><th>Problem</th><th>Sample</th><th>Layer</th><th>Status</th><th>Score</th></tr></thead>"
                f"<tbody>{details}</tbody></table></details></article>"
            )
        sections.append(
            f"<section><h2>Comparison group {index}</h2>"
            f"<p>Hardware: {escaped(contract['hardware'])}; precision: {escaped(contract['precision'])}; "
            f"level: {contract['level']}</p>"
            f"<details><summary>Experiment settings</summary><pre>{escaped(json.dumps(contract, indent=2))}</pre></details>"
            + "".join(members) + "</section>"
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>KernelBench Pro results</title><style>body{font:16px system-ui;max-width:1000px;margin:2em auto;padding:1em}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:.4em;text-align:left}"
        "section,article{margin:1.5em 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>KernelBench Pro results</h1><p>Each group shares declared workload, hardware and budget settings. "
        "Failures and missing samples remain in the denominator.</p>" + "".join(sections) + "</html>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Native run directories containing run_config.json")
    parser.add_argument("--output", "--out", type=Path, default=Path("leaderboard.html"))
    args = parser.parse_args(argv)
    result = render(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

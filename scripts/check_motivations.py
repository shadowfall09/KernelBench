"""Validate that every inventoried change has a motivation and local evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory_features(text):
    version, header, counts = 0, "", Counter()
    features = {}
    for line in text.splitlines():
        match = re.match(r"## 第([一二三四五六七八九十\d]+)版", line)
        if match:
            raw = match[1]
            digits = {name: i for i, name in enumerate("一二三四五六七八九", 1)}
            if raw.isdecimal():
                version = int(raw)
            elif "十" in raw:
                tens, units = raw.split("十")
                version = 10 * (digits[tens] if tens else 1) + (digits[units] if units else 0)
            else:
                version = digits[raw]
            header = ""
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in {"编号", "项目", "功能"}:
            header = cells[1]
            continue
        explicit = re.fullmatch(r"V\d+-\d+", cells[0])
        if explicit or (version == 6 and header == "当前变化" and not cells[0].startswith("-")) or (
                version == 7 and header == "相对第六版的变化" and not cells[0].startswith("-")):
            counts[version] += 1
            key = cells[0] if explicit else f"V{version}-{counts[version]:02}"
            if key in features:
                raise ValueError(f"Duplicate inventory feature: {key}")
            features[key] = cells[1] if explicit else cells[0]
    return features


def render_tables(data):
    changes, roadmap = [], ["| 设计 | 状态 | Motivation | 已交付与剩余范围 |", "| --- | --- | --- | --- |"]
    for motivation, title in data["motivations"].items():
        changes += [f"### {motivation}：{title}的改动", "", "| 改动 | 回应 | 关联 | 证据 |", "| --- | --- | --- | --- |"]
        for row in data["changes"]:
            if row["primary"] != motivation or row["status"] != "implemented" or not row["id"].startswith("V"):
                continue
            evidence = ", ".join(f"[{Path(path).name}](../{path})" for path in row["evidence"])
            changes.append(f"| {row['id']} {row['title']} | {row['reason']} | {', '.join(row['secondary']) or '—'} | {evidence} |")
        changes.append("")
    for row in data["changes"]:
        if row["id"].startswith("R-"):
            state = {"planned": "待实现", "in_progress": "部分实现", "implemented": "代码已实现"}[row["status"]]
            roadmap.append(f"| {row['id']} {row['title']} | {state} | {row['primary']} | {row['reason']} |")
    return {"changes": "\n".join(changes).strip(), "roadmap": "\n".join(roadmap)}


def validate(root=ROOT, write=False):
    root = Path(root)
    data = json.loads((root / "docs/motivation_map.json").read_text())
    indexed = inventory_features((root / "docs/FEATURE_INVENTORY.md").read_text())
    by_id = {row["id"]: row for row in data["changes"]}
    if len(by_id) != len(data["changes"]):
        raise ValueError("Duplicate motivation mapping IDs")
    implemented = {key: row for key, row in by_id.items() if row["status"] == "implemented" and key.startswith("V")}
    if set(indexed) != set(implemented):
        raise ValueError(f"Inventory/motivation coverage mismatch: {set(indexed) ^ set(implemented)}")
    for row in by_id.values():
        if (row["primary"] not in data["motivations"] or row["status"] not in {"implemented", "in_progress", "planned"}
                or any(key not in data["motivations"] or key == row["primary"] for key in row["secondary"])):
            raise ValueError(f"Invalid mapping: {row['id']}")
        if not row["reason"] or not row["evidence"]:
            raise ValueError(f"Missing rationale/evidence: {row['id']}")
        for name in row["evidence"]:
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Evidence unavailable in checkout: {name}")
        if row["id"].startswith("V") and row["status"] == "implemented" and indexed[row["id"]] != row["title"]:
            raise ValueError(f"Inventory title drift: {row['id']}")
    path = root / "docs/MOTIVATION.md"
    text = path.read_text()
    updated = text
    for key, table in render_tables(data).items():
        pattern = rf"(<!-- {key}:start -->)\n[\s\S]*?(<!-- {key}:end -->)"
        updated, count = re.subn(pattern, lambda match: f"{match[1]}\n{table}\n{match[2]}", updated)
        if count != 1:
            raise ValueError(f"Missing or duplicate table markers: {key}")
    if write:
        path.write_text(updated)
    elif updated != text:
        raise ValueError("Motivation tables are stale; run python -m scripts.check_motivations --write")
    designs = [row for row in by_id.values() if row["id"].startswith("R-")]
    return {"implemented_changes": len(implemented), "planned_designs": sum(row["status"] == "planned" for row in designs),
            "in_progress_designs": sum(row["status"] == "in_progress" for row in designs),
            "completed_designs": sum(row["status"] == "implemented" for row in designs),
            "historical_changes": sum(key.startswith(tuple(f"V{i}-" for i in range(1, 8))) for key in implemented),
            "paused_stash_excluded": data["paused_stash_excluded"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh the generated tables before validating")
    args = parser.parse_args()
    print(json.dumps(validate(write=args.write), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

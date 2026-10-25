"""Freeze saved candidates/jobs and run only current verifiers, with explicit execution."""

import argparse
import json
from pathlib import Path

from kb_workflows.common import read_json
from kb_workflows.config import parse_config
from kb_workflows.runtime import harbor_regrade, regrade


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name, source in (("native-plan", "source-run"), ("harbor-plan", "source-job")):
        action = actions.add_parser(name)
        action.add_argument("--" + source, type=Path, required=True)
        action.add_argument("--config", type=Path, required=True)
        action.add_argument("--output", type=Path, required=True)
    for name in ("native-run", "harbor-run"):
        action = actions.add_parser(name)
        action.add_argument("--manifest", type=Path, required=True)
        action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action.endswith("-plan"):
            config = parse_config(["--config", str(args.config), "--generation", "local", "--stage", "evaluate"])
            result = (regrade.prepare(args.source_run, config, output=args.output) if args.action == "native-plan"
                      else harbor_regrade.prepare(args.source_job, config, output=args.output))
            print(json.dumps({"experiments_started": False, "plan_sha256": result["plan_sha256"], "output": str(args.output)}, indent=2))
        else:
            module = regrade if args.action == "native-run" else harbor_regrade
            result = module.execute(read_json(args.manifest), execute=args.execute)
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate or explicitly execute a multi-session protocol study."""

import argparse
import json
from pathlib import Path

from kb_workflows.common import read_json
from kb_workflows.study_plan import execute_plan, make_plan, write_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--spec", type=Path)
    action.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="Start actual model and device experiments")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    if args.spec:
        if args.output_dir is None:
            parser.error("--spec requires --output-dir")
        plan = make_plan(read_json(args.spec), directory=args.output_dir, base_directory=args.spec.resolve().parent)
        path = write_plan(plan, args.output_dir)
    else:
        path = args.plan
    result = execute_plan(path, execute=args.execute, keep_going=args.keep_going)
    print(json.dumps(result, indent=2))
    return 1 if any(job["exit_code"] for job in result.get("jobs", [])) else 0


if __name__ == "__main__":
    raise SystemExit(main())

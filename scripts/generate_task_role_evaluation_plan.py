#!/usr/bin/env python3
"""Generate the fixed paired target and regression evaluation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.task_role_intervention import build_evaluation_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation plan: {args.output}")
    plans = build_evaluation_plan(yaml.safe_load(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in plans))
    print(json.dumps({"output": str(args.output), "trials": len(plans)}, sort_keys=True))


if __name__ == "__main__":
    main()

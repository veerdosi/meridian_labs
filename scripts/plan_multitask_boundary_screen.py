#!/usr/bin/env python3
"""Plan a preregistered multi-suite boundary screen after a negative intervention gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.multitask import build_multitask_screen_plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    plans_by_suite = build_multitask_screen_plans(config)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for suite, plans in plans_by_suite.items():
        path = args.output / f"{suite}.jsonl"
        with path.open("w") as stream:
            for plan in plans:
                stream.write(json.dumps(plan, sort_keys=True) + "\n")
        summary[suite] = {"rollouts": len(plans), "path": str(path)}
    summary["total_rollouts"] = sum(len(plans) for plans in plans_by_suite.values())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Nominate screen candidates and write paired multi-task validation plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.multitask import build_multitask_validation_plans, nominate_multitask_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-plans", type=Path, required=True)
    parser.add_argument("--output-nominations", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    results = []
    for path in args.results:
        results.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    candidates = nominate_multitask_candidates(results, config)
    plans = build_multitask_validation_plans(candidates, config)
    args.output_plans.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plans.open("w") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, sort_keys=True) + "\n")
    args.output_nominations.parent.mkdir(parents=True, exist_ok=True)
    args.output_nominations.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"nominees": len(candidates), "validation_rollouts": len(plans)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Nominate semantic discovery failures and write locked repeated validation plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.semantic_discovery import (
    build_semantic_validation_plans,
    nominate_validation_candidates,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-tasks", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    results = [row for path in args.results for row in load_jsonl(path)]
    candidates = nominate_validation_candidates(
        results, load_jsonl(args.discovery_tasks), config
    )
    plans = build_semantic_validation_plans(candidates, config)
    args.output.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (args.output / "candidates.jsonl", candidates),
        (args.output / "validation_plans.jsonl", plans),
    ):
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"candidates": len(candidates), "validation_rollouts": len(plans)}))


if __name__ == "__main__":
    main()

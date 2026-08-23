#!/usr/bin/env python3
"""Evaluate the locked harder-boundary confirmation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.confirmation import evaluate_harder_boundary_confirmation
from meridian.gating import merge_rollouts


def load(paths: list[Path]) -> list[dict]:
    groups = [
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for path in paths
    ]
    return merge_rollouts(*groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--released", type=Path, action="append", required=True)
    parser.add_argument("--targeted", type=Path, action="append", required=True)
    parser.add_argument("--random", type=Path, action="append", required=True)
    parser.add_argument("--original", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    decision = evaluate_harder_boundary_confirmation(
        released=load(args.released),
        targeted=load(args.targeted),
        random=load(args.random),
        original=load(args.original),
        thresholds=config["confirmation_thresholds"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()

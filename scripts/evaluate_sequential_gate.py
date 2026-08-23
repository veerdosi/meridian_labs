#!/usr/bin/env python3
"""Apply the locked targeted-vs-random gate to complete paired rollout bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meridian.gating import evaluate_sequential_gate, merge_rollouts


def load(paths: list[Path]) -> list[dict]:
    return merge_rollouts(
        *[
            [json.loads(line) for line in path.read_text().splitlines() if line]
            for path in paths
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--targeted", type=Path, action="append", required=True)
    parser.add_argument("--random", type=Path, action="append", required=True)
    parser.add_argument("--regression-limit", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = evaluate_sequential_gate(
        baseline=load(args.baseline),
        targeted=load(args.targeted),
        random=load(args.random),
        regression_limit=args.regression_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True))
    print(json.dumps({"gate": decision["gate"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

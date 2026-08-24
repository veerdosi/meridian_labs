#!/usr/bin/env python3
"""Evaluate one locked physical-boundary dose stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meridian.gating import evaluate_dose_gate, merge_rollouts


def load(paths: list[Path] | None) -> list[dict] | None:
    if not paths:
        return None
    return merge_rollouts(*[[json.loads(line) for line in path.read_text().splitlines() if line] for path in paths])


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in ("baseline", "targeted", "random"):
        parser.add_argument(f"--{arm}", type=Path, action="append", required=True)
    parser.add_argument("--original", type=Path, action="append")
    parser.add_argument("--prior-decision", type=Path)
    parser.add_argument("--dose", type=int, required=True)
    parser.add_argument("--regression-limit", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior_decision.read_text())["arms"]["targeted"] if args.prior_decision else None
    result = evaluate_dose_gate(
        baseline=load(args.baseline), targeted=load(args.targeted), random=load(args.random),
        original=load(args.original), regression_limit=args.regression_limit, dose=args.dose,
        prior_targeted=prior,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

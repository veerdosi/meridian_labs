#!/usr/bin/env python3
"""Score repeated boundary validation under the locked selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.multitask import score_validated_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nominations", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    candidates = json.loads(args.nominations.read_text())
    results = []
    for path in args.results:
        results.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ranking = score_validated_candidates(results, candidates, config)
    eligible = [row for row in ranking if row["eligible"]]
    decision = {
        "rule": "highest-ranked eligible candidate; no discretionary override",
        "selected": eligible[0] if eligible else None,
        "training_permitted": bool(eligible),
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"eligible": len(eligible), "selected": decision["selected"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select a validated boundary strictly under the locked semantic rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.semantic_discovery import rank_validated_boundaries


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    results = [row for path in args.results for row in load_jsonl(path)]
    ranking = rank_validated_boundaries(results, load_jsonl(args.candidates), config)
    eligible = [row for row in ranking if row["eligible"]]
    decision = {
        "rule": "highest-ranked eligible boundary; no discretionary override",
        "selected": eligible[0] if eligible else None,
        "training_permitted": bool(eligible),
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"eligible": len(eligible), "selected": decision["selected"]}))


if __name__ == "__main__":
    main()

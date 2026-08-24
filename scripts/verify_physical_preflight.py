#!/usr/bin/env python3
"""Verify the one-trajectory physical telemetry engineering preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meridian.rollout_integrity import verify_physical_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-id", default="physical-telemetry-preflight")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()]
    if len(records) != 1 or records[0]["id"] != args.expected_id:
        raise ValueError("preflight bundle does not contain the one exact expected rollout")
    result = verify_physical_rollout(records[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create compact, evidence-linked physical stage summaries from rollout JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meridian.trajectory_diagnosis import diagnose_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for path in args.rollouts
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("no rollout records")
    diagnoses = [diagnose_record(record) for record in records]
    payload = {
        "schema": "meridian-physical-stage-diagnosis-v2",
        "records": diagnoses,
        "limitations": [
            "stage labels localize observed behavior but do not identify a unique root cause",
            "a data-coverage claim requires repeatability, geometry clustering, and control probes",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"records": len(diagnoses), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

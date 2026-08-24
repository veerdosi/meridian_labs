#!/usr/bin/env python3
"""Evaluate a completed role-binding confirmation batch against its pre-locked gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.task_role_boundary import assess_role_confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--diagnoses", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rollouts = [
        json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()
    ]
    diagnoses = json.loads(args.diagnoses.read_text())["records"]
    config = yaml.safe_load(args.config.read_text())
    result = assess_role_confirmation(rollouts, diagnoses, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({key: result[key] for key in ("automated_pass", "decision")}, sort_keys=True))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Materialize the locked outcome-blind screening plan and compact lock manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.physical_boundary import make_screening_plans, validate_protocol_config
from meridian.rollout_integrity import canonical_sha256, reserve_results_path, validate_plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_protocol_config(config)
    plans = validate_plans(make_screening_plans(config))
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "screening.jsonl"
    reserve_results_path(plan_path)
    with plan_path.open("a") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, sort_keys=True) + "\n")
    manifest = {
        "schema": config["schema"],
        "config_sha256": canonical_sha256(config),
        "screening_plan_sha256": canonical_sha256({"plans": plans}),
        "screening_trials": len(plans),
        "untouched_holdout_init_states": config["partitions"]["untouched_holdout_init_states"],
        "training_authorized": False,
    }
    (args.output / "lock.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

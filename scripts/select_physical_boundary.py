#!/usr/bin/env python3
"""Apply the locked screen/confirmation rule without authorizing training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.physical_boundary import (
    build_physical_capability_map,
    finalize_selection,
    make_confirmation_plans,
    make_evaluation_plans,
    make_source_pool_plans,
    summarize_screening,
    validate_protocol_config,
)
from meridian.rollout_integrity import reserve_results_path, validate_plans


def load_jsonl(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--screening", type=Path, action="append", required=True)
    parser.add_argument("--diagnoses", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, action="append")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_protocol_config(config)
    diagnoses = {item["id"]: item for item in json.loads(args.diagnoses.read_text())["records"]}
    screening_records = load_jsonl(args.screening)
    summaries = summarize_screening(screening_records, diagnoses, config)
    args.output.mkdir(parents=True, exist_ok=True)
    capability_map = build_physical_capability_map(summaries, screening_records)
    (args.output / "capability-map.json").write_text(
        json.dumps(capability_map, indent=2, sort_keys=True)
    )
    if not args.confirmation:
        plans = validate_plans(
            make_confirmation_plans(summaries, load_jsonl([args.inventory]), config)
        )
        path = args.output / "confirmation.jsonl"
        reserve_results_path(path)
        with path.open("a") as stream:
            for plan in plans:
                stream.write(json.dumps(plan, sort_keys=True) + "\n")
        result = {"phase": "confirmation_required", "plans": len(plans), "training_authorized": False}
    else:
        result = finalize_selection(
            summaries,
            load_jsonl(args.confirmation),
            load_jsonl([args.inventory]),
            config,
        )
        if result["selected"]:
            source_plans = validate_plans(make_source_pool_plans(result, config))
            evaluation_plans = validate_plans(
                make_evaluation_plans(result, load_jsonl([args.inventory]), screening_records, config)
            )
            for name, plans in (("source-pool.jsonl", source_plans), ("evaluation.jsonl", evaluation_plans)):
                path = args.output / name
                reserve_results_path(path)
                with path.open("a") as stream:
                    for plan in plans:
                        stream.write(json.dumps(plan, sort_keys=True) + "\n")
            result["source_pool_plans"] = len(source_plans)
            result["evaluation_plans"] = len(evaluation_plans)
        (args.output / "selection.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select unique nested targeted/random/original source records after source collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.data_selection import select_data_arms
from meridian.physical_boundary import validate_protocol_config
from meridian.rollout_integrity import reserve_results_path


def load_jsonl(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--boundary-rollouts", type=Path, action="append", required=True)
    parser.add_argument("--successful-pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_protocol_config(config)
    selection = json.loads(args.selection.read_text())
    boundary_records = load_jsonl(args.boundary_rollouts)
    boundary_id = selection["selected"]["boundary_rollout_id"]
    boundary = next(record for record in boundary_records if record["id"] == boundary_id)
    pool = load_jsonl(args.successful_pool)
    manifest = select_data_arms(selection, boundary, pool, config)
    by_id = {record["id"]: record for record in pool}
    args.output.mkdir(parents=True, exist_ok=True)
    for dose, arms in manifest["doses"].items():
        for arm, identifiers in arms.items():
            path = args.output / f"dose-{dose}-{arm}.jsonl"
            reserve_results_path(path)
            with path.open("a") as stream:
                for identifier in identifiers:
                    stream.write(json.dumps(by_id[identifier], sort_keys=True) + "\n")
    (args.output / "data-selection.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

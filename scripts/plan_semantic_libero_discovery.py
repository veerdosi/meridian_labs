#!/usr/bin/env python3
"""Inventory LIBERO semantics and lock discovery/reserve task splits and plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.semantic_discovery import (
    build_discovery_plans,
    inventory_tasks,
    score_inventory,
    select_discovery_and_reserve,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-map", type=Path, required=True)
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    inventory = inventory_tasks(args.task_map, args.bddl_root, config["compatible_suites"])
    scored = score_inventory(inventory, config)
    discovery, reserve = select_discovery_and_reserve(scored, config)
    plans = build_discovery_plans(discovery, config)
    write_jsonl(args.output / "inventory.jsonl", scored)
    write_jsonl(args.output / "discovery_tasks.jsonl", discovery)
    write_jsonl(args.output / "confirmation_reserve_tasks.jsonl", reserve)
    write_jsonl(args.output / "discovery_plans.jsonl", plans)
    summary = {
        "inventory": len(scored),
        "discovery_tasks": len(discovery),
        "confirmation_reserve_tasks": len(reserve),
        "discovery_rollouts": len(plans),
        "discovery_keys": [[row["task_suite"], row["task_id"]] for row in discovery],
        "confirmation_reserve_keys": [[row["task_suite"], row["task_id"]] for row in reserve],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

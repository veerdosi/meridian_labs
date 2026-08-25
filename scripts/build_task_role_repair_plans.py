#!/usr/bin/env python3
"""Materialize the locked nested targeted and random simulator-expert plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from meridian.task_role_repair import (
    build_expert_development_plans,
    build_expert_validation_plans,
    build_random_plans,
    build_targeted_plans,
    validate_repair_config,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = args.config.resolve().parents[1]
    validate_repair_config(config, root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for phase, records in (
        ("expert-development", build_expert_development_plans(config)),
        ("expert-validation", build_expert_validation_plans(config)),
    ):
        path = args.output_dir / f"{phase}.jsonl"
        write_jsonl(path, records)
        outputs.append(
            {
                "arm": phase,
                "episodes": len(records),
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    for dose in map(int, config["selection"]["doses_new_episodes"]):
        for arm, builder in (("targeted", build_targeted_plans), ("random", build_random_plans)):
            path = args.output_dir / f"{arm}-dose{dose}.jsonl"
            records = builder(config, dose)
            write_jsonl(path, records)
            outputs.append(
                {
                    "arm": arm,
                    "dose": dose,
                    "episodes": len(records),
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
    manifest = {
        "schema": "task-role-repair-plans-v1",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "bddl_inputs": [
            {
                "path": str(variant["bddl"]),
                "sha256": file_sha256(root / str(variant["bddl"])),
            }
            for task in config["tasks"]
            for variant in task["role_variants"]
        ],
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

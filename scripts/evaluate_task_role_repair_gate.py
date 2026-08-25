#!/usr/bin/env python3
"""Apply the locked contrastive repair gate to paired completed evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meridian.task_role_repair import assess_repair_gate


def load_arm(path: Path, arm: str) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    normalized = []
    for record in records:
        parameters = record.get("parameters", {})
        suite = record.get("evaluation_suite", parameters.get("evaluation_suite"))
        if suite not in {"target", "regression"}:
            raise ValueError(f"rollout lacks a target/regression label: {record.get('id')}")
        normalized.append(
            {
                "arm": arm,
                "id": str(record["id"]),
                "evaluation_suite": suite,
                "task_suite": str(record["task_suite"]),
                "task_id": int(record["task_id"]),
                "success": bool(record["success"]),
            }
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dose", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    records = (
        load_arm(args.baseline, "baseline")
        + load_arm(args.targeted, "targeted")
        + load_arm(args.random, "random")
    )
    assessment = assess_repair_gate(records, config)
    assessment["dose"] = args.dose
    successes = assessment["successes"]
    regression = assessment["regression_successes"]
    if args.dose == min(map(int, config["selection"]["doses_new_episodes"])):
        if successes["targeted"] <= successes["random"]:
            decision = "stop_negative"
            rationale = "targeted did not strictly beat equal-dose random"
        elif regression["targeted"] < regression["baseline"]:
            decision = "stop_negative"
            rationale = "targeted lost a locked regression trial"
        else:
            decision = "release_medium"
            rationale = "targeted beat random with clean regression; estimate marginal return"
    else:
        decision = "accept_repair" if assessment["decisive_pass"] else "reject_repair"
        rationale = (
            "all locked decisive checks passed"
            if assessment["decisive_pass"]
            else "one or more locked decisive checks failed"
        )
    assessment["decision"] = decision
    assessment["rationale"] = rationale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(assessment, sort_keys=True))


if __name__ == "__main__":
    main()

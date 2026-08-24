#!/usr/bin/env python3
"""Report locked semantic validation outcomes without changing the selection rule."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from meridian.gating import _paired_sign_test
from meridian.search import _wilson

CONDITIONS = (
    "canonical",
    "compound_view_visual_canonical_control",
    "viewpoint_only",
    "visual_only",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict], decision: dict, campaign: dict) -> dict:
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (str(row["task_suite"]), int(row["task_id"]))
        condition = str(row["parameters"]["validation_condition"])
        grouped[key][condition].append(row)

    candidates = []
    for key in sorted(grouped):
        by_condition = grouped[key]
        if set(by_condition) != set(CONDITIONS):
            raise ValueError(f"incomplete conditions for {key}: {sorted(by_condition)}")
        condition_summary = {}
        paired_outcomes = {}
        outcomes = {}
        for condition in CONDITIONS:
            members = by_condition[condition]
            successes = sum(bool(row["success"]) for row in members)
            condition_summary[condition] = {
                "successes": successes,
                "trials": len(members),
                "success_rate": successes / len(members),
                "wilson_95": _wilson(successes, len(members)),
            }
            outcomes[condition] = {
                (int(row["seed"]), float(row["parameters"]["init_state_index"])): bool(
                    row["success"]
                )
                for row in members
            }
        for condition in CONDITIONS[1:]:
            paired_outcomes[f"canonical_vs_{condition}"] = _paired_sign_test(
                outcomes["canonical"], outcomes[condition]
            )
        candidates.append(
            {
                "task_suite": key[0],
                "task_id": key[1],
                "conditions": condition_summary,
                "paired_outcomes": paired_outcomes,
            }
        )

    attempts = campaign["validation_execution"]["attempts"]
    return {
        "campaign_id": "pi05-libero-semantic-validation-v1",
        "scientific_plan_sha256": campaign["validation_execution"]["plan_sha256"],
        "rollouts": len(rows),
        "candidates": candidates,
        "locked_decision": decision,
        "pbs_attempts": attempts,
        "actual_campaign_su": campaign["accounting"]["campaign_su_to_date"],
        "actual_cumulative_measured_su": campaign["accounting"]["cumulative_measured_su"],
        "benchmark_results_used_as_confirmation_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        load_jsonl(args.results),
        json.loads(args.decision.read_text()),
        yaml.safe_load(args.campaign_manifest.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidates": len(payload["candidates"]), "rollouts": payload["rollouts"]}))


if __name__ == "__main__":
    main()

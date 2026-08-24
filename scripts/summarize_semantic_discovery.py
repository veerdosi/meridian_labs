#!/usr/bin/env python3
"""Summarize paired semantic discovery outcomes and uncertainty."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from meridian.search import _wilson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--walltime-seconds", type=float, required=True)
    parser.add_argument("--su", type=float, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    candidates = [
        json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    task_pairs: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        profile = str(row["parameters"]["stress_profile"])
        groups[(str(row["task_suite"]), profile)].append(row)
        task_pairs[(str(row["task_suite"]), int(row["task_id"]))][profile] = bool(
            row["success"]
        )
    suite_profiles = {}
    for (suite, profile), members in sorted(groups.items()):
        successes = sum(bool(row["success"]) for row in members)
        suite_profiles.setdefault(suite, {})[profile] = {
            "successes": successes,
            "trials": len(members),
            "success_rate": successes / len(members),
            "wilson_95": _wilson(successes, len(members)),
        }
    payload = {
        "campaign_id": "pi05-libero-semantic-discovery-v1",
        "job": {
            "job_id": args.job_id,
            "walltime_seconds": args.walltime_seconds,
            "su": args.su,
            "exit_status": 0,
        },
        "rollouts": len(rows),
        "suite_profiles": suite_profiles,
        "screen_qualifiers": [
            {"task_suite": suite, "task_id": task_id}
            for (suite, task_id), outcomes in sorted(task_pairs.items())
            if outcomes.get("canonical") is True
            and outcomes.get("compound_view_visual") is False
        ],
        "locked_validation_nominees": candidates,
        "selection_status": "nominees_only_pending_repeated_validation",
        "training_permitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rollouts": len(rows), "nominees": len(candidates)}))


if __name__ == "__main__":
    main()

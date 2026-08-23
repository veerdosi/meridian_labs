#!/usr/bin/env python3
"""Turn observed failures into matched, low-cost discriminating rollout probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CANONICAL = {
    "camera_x": 0.0,
    "camera_yaw_deg": 0.0,
    "brightness": 1.0,
    "occlusion": 0.0,
    "visual_distractors": 0.0,
    "replan_steps": 5.0,
    "action_noise": 0.0,
}
VIEW = {"camera_x", "camera_yaw_deg"}
VISUAL = {"brightness", "occlusion", "visual_distractors"}
CONTROL = {"replan_steps", "action_noise"}


def condition(source: dict, family: str, retained: set[str]) -> dict:
    parameters = {**CANONICAL, "init_state_index": source["parameters"]["init_state_index"]}
    for key in retained:
        parameters[key] = source["parameters"][key]
    return {
        "id": f"{source['id']}-probe-{family}",
        "task_suite": source["task_suite"],
        "task_id": source["task_id"],
        "seed": source["seed"],
        "probe_family": family,
        "source_failure_id": source["id"],
        **parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stochastic-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--failure-id", action="append")
    parser.add_argument("--repeat-family", action="append")
    parser.add_argument("--only-repeats", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line]
    failures = [record for record in records if not record["success"]]
    if args.failure_id:
        selected_ids = set(args.failure_id)
        failures = [record for record in failures if record["id"] in selected_ids]
    if not failures:
        raise SystemExit("no failures found")

    plans = []
    families = {
        "exact": VIEW | VISUAL | CONTROL,
        "canonical": set(),
        "view-only": VIEW,
        "visual-only": VISUAL,
        "control-only": CONTROL,
        "view-visual": VIEW | VISUAL,
        "view-visual-action-noise": VIEW | VISUAL | {"action_noise"},
        "view-visual-replan": VIEW | VISUAL | {"replan_steps"},
    }
    repeat_families = args.repeat_family or ["exact"]
    unknown = set(repeat_families) - families.keys()
    if unknown:
        raise SystemExit(f"unknown repeat families: {', '.join(sorted(unknown))}")
    for failure_index, failure in enumerate(failures):
        if not args.only_repeats:
            plans.extend(condition(failure, name, retained) for name, retained in families.items())
        for family_index, family in enumerate(repeat_families):
            for repeat in range(args.stochastic_repeats):
                plan = condition(failure, f"{family}-repeat-{repeat}", families[family])
                plan["seed"] = (
                    args.seed
                    + failure_index * len(repeat_families) * args.stochastic_repeats
                    + family_index * args.stochastic_repeats
                    + repeat
                )
                plans.append(plan)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, sort_keys=True) + "\n")
    print(f"failures={len(failures)} probes={len(plans)} output={args.output}")


if __name__ == "__main__":
    main()

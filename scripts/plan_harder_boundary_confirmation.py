#!/usr/bin/env python3
"""Resolve the locked untouched holdouts for the selected harder boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

CANONICAL = {
    "camera_x": 0.0,
    "camera_yaw_deg": 0.0,
    "brightness": 1.0,
    "occlusion": 0.0,
    "visual_distractors": 0.0,
    "replan_steps": 5.0,
    "action_noise": 0.0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.confirmation.read_text())
    decision = json.loads(args.selection.read_text())
    selected = decision.get("selected")
    if not decision.get("training_permitted") or not selected or not selected.get("eligible"):
        raise SystemExit("locked validation gate did not permit confirmation or training")

    target = spec["untouched_holdouts"]["target"]
    bounds = target["profile_bounds"][selected["candidate_profile"]]
    rng = np.random.default_rng(int(target["perturbation_jitter_seed"]))
    plans = []
    for index, seed in enumerate(target["seeds"]):
        point = {
            name: float(low) if low == high else float(rng.uniform(low, high))
            for name, (low, high) in bounds.items()
        }
        plans.append(
            {
                "id": f"harder-target-{index:03d}",
                "task_suite": selected["task_suite"],
                "task_id": int(selected["task_id"]),
                "seed": int(seed),
                "init_state_index": float((7 + 17 * index) % 50),
                "evaluation_suite": "target",
                "candidate_profile": selected["candidate_profile"],
                **CANONICAL,
                **point,
            }
        )
    regression = spec["untouched_holdouts"]["regression"]
    for suite in regression["suites"]:
        for index, seed in enumerate(regression["seeds_by_suite"][suite]):
            plans.append(
                {
                    "id": f"harder-regression-{suite}-{index:02d}",
                    "task_suite": suite,
                    "task_id": (int(selected["task_id"]) + index) % 10,
                    "seed": int(seed),
                    "init_state_index": float((13 + 19 * index) % 50),
                    "evaluation_suite": f"regression:{suite}",
                    **CANONICAL,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, sort_keys=True) + "\n")
    print(json.dumps({"target": len(target["seeds"]), "regression": 40, "total": len(plans)}))


if __name__ == "__main__":
    main()

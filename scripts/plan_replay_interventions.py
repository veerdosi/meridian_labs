#!/usr/bin/env python3
"""Create matched real-simulator replay plans from intervention specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from meridian.models import ExperimentSpec, InterventionArm, InterventionSpec

OBSERVATION_AXES = {
    "camera_x",
    "camera_yaw_deg",
    "brightness",
    "occlusion",
    "visual_distractors",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument("--source-rollouts", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = ExperimentSpec.model_validate(yaml.safe_load(args.config.read_text()))
    payload = yaml.safe_load(args.interventions.read_text())
    interventions = [InterventionSpec.model_validate(item) for item in payload["interventions"]]
    records = []
    for path in args.source_rollouts:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    sources = [record for record in records if record["success"]]
    if not sources:
        raise SystemExit("no successful source rollouts")
    canonical = {axis.name: axis.canonical for axis in spec.parameter_space.axes}
    if any(value is None for value in canonical.values()):
        raise SystemExit("all axes require canonical values")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for intervention in interventions:
        if intervention.arm in {InterventionArm.NONE, InterventionArm.NO_DATA_FIX}:
            continue
        rng = np.random.default_rng(intervention.seed)
        eligible = sources
        init_bounds = intervention.target_bounds.get("init_state_index")
        if intervention.arm in {InterventionArm.TARGETED, InterventionArm.ORACLE} and init_bounds:
            eligible = [
                source
                for source in sources
                if init_bounds[0] <= source["init_state_index"] <= init_bounds[1]
            ]
        if not eligible:
            raise SystemExit(f"no eligible source rollout for {intervention.id}")
        order = rng.permutation(len(eligible))
        plans = []
        for index in range(intervention.trajectory_count):
            source = eligible[int(order[index % len(order)])]
            point = dict(canonical)
            point["init_state_index"] = float(source["init_state_index"])
            for axis in spec.parameter_space.axes:
                if axis.name not in OBSERVATION_AXES:
                    continue
                if intervention.arm == InterventionArm.ORIGINAL:
                    value = canonical[axis.name]
                elif intervention.arm == InterventionArm.RANDOM:
                    value = float(rng.uniform(axis.low, axis.high))
                elif intervention.arm == InterventionArm.ORACLE:
                    low, high = intervention.target_bounds[axis.name]
                    value = float((low + high) / 2)
                else:
                    low, high = intervention.target_bounds[axis.name]
                    value = float(rng.uniform(low, high))
                point[axis.name] = value
            plans.append(
                {
                    "id": f"{intervention.id}-episode-{index:04d}",
                    "task_suite": source["task_suite"],
                    "task_id": source["task_id"],
                    "seed": intervention.seed * 1000 + index,
                    "source_rollout_id": source["id"],
                    "intervention_id": intervention.id,
                    "arm": intervention.arm.value,
                    **point,
                }
            )
        output = args.output / f"{intervention.arm.value}.jsonl"
        with output.open("w") as stream:
            for plan in plans:
                stream.write(json.dumps(plan, sort_keys=True) + "\n")
        summary[intervention.arm.value] = {
            "intervention_id": intervention.id,
            "plans": len(plans),
            "unique_sources": len({plan["source_rollout_id"] for plan in plans}),
            "path": str(output),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

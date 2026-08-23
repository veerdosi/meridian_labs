#!/usr/bin/env python3
"""Create matched real-simulator replay plans from intervention specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from meridian.intervention import sample_axis_value
from meridian.models import (
    ExperimentSpec,
    InterventionArm,
    InterventionSamplingStrategy,
    InterventionSpec,
)

OBSERVATION_AXES = {
    "camera_x",
    "camera_yaw_deg",
    "brightness",
    "occlusion",
    "visual_distractors",
}


def resolved_strategy(intervention: InterventionSpec) -> InterventionSamplingStrategy:
    if intervention.sampling_strategy != InterventionSamplingStrategy.ARM_DEFAULT:
        return intervention.sampling_strategy
    return {
        InterventionArm.TARGETED: InterventionSamplingStrategy.BOUNDS_UNIFORM,
        InterventionArm.ORACLE: InterventionSamplingStrategy.CENTER,
        InterventionArm.RANDOM: InterventionSamplingStrategy.FULL_UNIFORM,
        InterventionArm.ORIGINAL: InterventionSamplingStrategy.CANONICAL,
    }[intervention.arm]


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
    used_source_ids = set()
    for intervention in interventions:
        if intervention.arm in {InterventionArm.NONE, InterventionArm.NO_DATA_FIX}:
            continue
        rng = np.random.default_rng(intervention.seed)
        strategy = resolved_strategy(intervention)
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
        target_components: list[bool | None] = [None] * intervention.trajectory_count
        if strategy == InterventionSamplingStrategy.EVIDENCE_WEIGHTED_MIXTURE:
            target_count = round(intervention.trajectory_count * intervention.target_fraction)
            flags = np.array(
                [True] * target_count
                + [False] * (intervention.trajectory_count - target_count)
            )
            target_components = [bool(value) for value in rng.permutation(flags)]
        plans = []
        for index in range(intervention.trajectory_count):
            source = eligible[int(order[index % len(order)])]
            point = dict(canonical)
            point["init_state_index"] = float(source["init_state_index"])
            use_target_component = target_components[index]
            for axis in spec.parameter_space.axes:
                if axis.name not in OBSERVATION_AXES:
                    continue
                low, high = intervention.target_bounds[axis.name]
                point[axis.name] = sample_axis_value(
                    strategy=strategy,
                    rng=rng,
                    axis_low=axis.low,
                    axis_high=axis.high,
                    target_low=low,
                    target_high=high,
                    canonical=float(canonical[axis.name]),
                    target_fraction=intervention.target_fraction,
                    use_target_component=use_target_component,
                )
            plans.append(
                {
                    "id": f"{intervention.id}-episode-{index:04d}",
                    "task_suite": source["task_suite"],
                    "task_id": source["task_id"],
                    "seed": intervention.seed * 1000 + index,
                    "source_rollout_id": source["id"],
                    "intervention_id": intervention.id,
                    "arm": intervention.arm.value,
                    "sampling_component": (
                        "target" if use_target_component else "broad"
                        if use_target_component is not None
                        else strategy.value
                    ),
                    **point,
                }
            )
            used_source_ids.add(source["id"])
        output = args.output / f"{intervention.arm.value}.jsonl"
        with output.open("w") as stream:
            for plan in plans:
                stream.write(json.dumps(plan, sort_keys=True) + "\n")
        summary[intervention.arm.value] = {
            "intervention_id": intervention.id,
            "plans": len(plans),
            "sampling_strategy": strategy.value,
            "target_fraction": intervention.target_fraction,
            "target_component_plans": sum(
                plan["sampling_component"] == "target" for plan in plans
            ),
            "unique_sources": len({plan["source_rollout_id"] for plan in plans}),
            "path": str(output),
        }
    source_output = args.output / "sources.jsonl"
    with source_output.open("w") as stream:
        for source in sources:
            if source["id"] in used_source_ids:
                stream.write(json.dumps(source, sort_keys=True) + "\n")
    summary["sources"] = {
        "records": len(used_source_ids),
        "path": str(source_output),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

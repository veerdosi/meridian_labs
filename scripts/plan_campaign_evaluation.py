#!/usr/bin/env python3
"""Create one fixed target and regression plan shared by all intervention arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from meridian.models import ExperimentSpec
from meridian.planning import spread_init_state_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-per-seed", type=int, default=4)
    parser.add_argument("--regression-per-suite", type=int, default=3)
    args = parser.parse_args()

    spec = ExperimentSpec.model_validate(yaml.safe_load(args.config.read_text()))
    campaign = yaml.safe_load(args.campaign.read_text())
    bounds = campaign["target_capability"]["bounds"]
    canonical = {axis.name: axis.canonical for axis in spec.parameter_space.axes}
    if any(value is None for value in canonical.values()):
        raise SystemExit("all axes require canonical values")

    plans = []
    for evaluation_seed in spec.evaluation_seeds:
        rng = np.random.default_rng(evaluation_seed)
        for repeat in range(args.target_per_seed):
            point = {name: float(rng.uniform(low, high)) for name, (low, high) in bounds.items()}
            plans.append(
                {
                    "id": f"target-seed{evaluation_seed}-repeat{repeat}",
                    "task_suite": spec.task_suite,
                    "task_id": spec.task_id,
                    "seed": evaluation_seed * 100 + repeat,
                    "evaluation_suite": "target",
                    "evaluation_seed": evaluation_seed,
                    **point,
                }
            )
    regression_suites = [name.removesuffix("_canonical") for name in spec.regression_suites]
    for suite_index, suite in enumerate(regression_suites):
        for repeat in range(args.regression_per_suite):
            plans.append(
                {
                    "id": f"regression-{suite}-repeat{repeat}",
                    "task_suite": suite,
                    "task_id": 0,
                    "seed": 9000 + suite_index * 100 + repeat,
                    "evaluation_suite": f"regression:{suite}",
                    "evaluation_seed": 9000 + suite_index,
                    **canonical,
                    "init_state_index": float(spread_init_state_index(repeat)),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "target": len(spec.evaluation_seeds) * args.target_per_seed,
                "regression": len(regression_suites) * args.regression_per_suite,
                "total": len(plans),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

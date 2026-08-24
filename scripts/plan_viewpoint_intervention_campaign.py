#!/usr/bin/env python3
"""Write the locked source, primary, and confirmation plans for task-3 intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from meridian.planning import spread_init_state_index


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def canonical(config: dict) -> dict[str, float]:
    return {key: float(value) for key, value in config["canonical_parameters"].items()}


def target_point(config: dict, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    point = canonical(config)
    for axis, bounds in config["target_viewpoint_bounds"].items():
        point[axis] = float(rng.uniform(*bounds))
    return point


def build_plans(config: dict) -> dict[str, list[dict]]:
    suite = str(config["target"]["task_suite"])
    task_id = int(config["target"]["task_id"])
    source = []
    for index, init_state in enumerate(config["training_source_pool"]["init_state_indices"]):
        source.append(
            {
                "id": f"viewpoint-v1-training-source-{index:03d}",
                "task_suite": suite,
                "task_id": task_id,
                "seed": int(config["training_source_pool"]["seed_base"]) + index,
                "evaluation_suite": "training_source",
                **canonical(config),
                "init_state_index": float(init_state),
            }
        )

    primary_target = []
    primary_task_regression = []
    primary = config["primary_evaluation"]
    primary_inits = list(primary["init_state_indices"])
    for seed_index, evaluation_seed in enumerate(primary["seeds"]):
        for repeat in range(int(primary["target_repeats_per_seed"])):
            plan_seed = int(evaluation_seed) * 100 + repeat
            primary_target.append(
                {
                    "id": f"primary-target-seed{evaluation_seed}-repeat{repeat}",
                    "task_suite": suite,
                    "task_id": task_id,
                    "seed": plan_seed,
                    "evaluation_seed": int(evaluation_seed),
                    "evaluation_suite": "target",
                    **target_point(config, plan_seed),
                    "init_state_index": float(
                        primary_inits[(seed_index + 3 * repeat) % len(primary_inits)]
                    ),
                }
            )
        for repeat in range(int(primary["task_regression_repeats_per_seed"])):
            primary_task_regression.append(
                {
                    "id": f"primary-regression-task3-seed{evaluation_seed}-repeat{repeat}",
                    "task_suite": suite,
                    "task_id": task_id,
                    "seed": int(evaluation_seed) * 100 + 50 + repeat,
                    "evaluation_seed": int(evaluation_seed),
                    "evaluation_suite": "regression:libero_10_task3_canonical",
                    **canonical(config),
                    "init_state_index": float(
                        primary_inits[(seed_index + 5 * repeat) % len(primary_inits)]
                    ),
                }
            )

    cross_regression = []
    for suite_index, regression_suite in enumerate(primary["cross_suite_regression"]):
        for repeat in range(int(primary["cross_suite_repeats"])):
            cross_regression.append(
                {
                    "id": f"primary-regression-{regression_suite}-repeat{repeat}",
                    "task_suite": regression_suite,
                    "task_id": 0,
                    "seed": int(primary["cross_suite_seed_base"]) + suite_index * 100 + repeat,
                    "evaluation_seed": int(primary["cross_suite_seed_base"]) + suite_index,
                    "evaluation_suite": f"regression:{regression_suite}_task0_canonical",
                    **canonical(config),
                    "init_state_index": float(spread_init_state_index(repeat)),
                }
            )

    confirmation = []
    confirm = config["confirmation_evaluation"]
    confirm_inits = list(confirm["init_state_indices"])
    for seed_index, evaluation_seed in enumerate(confirm["seeds"]):
        for repeat in range(int(confirm["target_repeats_per_seed"])):
            plan_seed = int(evaluation_seed) * 100 + repeat
            confirmation.append(
                {
                    "id": f"confirmation-target-seed{evaluation_seed}-repeat{repeat}",
                    "task_suite": suite,
                    "task_id": task_id,
                    "seed": plan_seed,
                    "evaluation_seed": int(evaluation_seed),
                    "evaluation_suite": "confirmation_target",
                    **target_point(config, plan_seed),
                    "init_state_index": float(
                        confirm_inits[(seed_index + 3 * repeat) % len(confirm_inits)]
                    ),
                }
            )

    primary_plans = primary_target + primary_task_regression + cross_regression
    if len(source) != int(config["training_source_pool"]["count"]):
        raise ValueError("training source count does not match its locked index list")
    if len(primary_target) != 40 or len(primary_plans) != 80 or len(confirmation) != 40:
        raise ValueError("locked campaign requires 40 target, 40 regression, 40 confirmation")
    source_seeds = {row["seed"] for row in source}
    primary_seeds = {row["seed"] for row in primary_plans}
    confirmation_seeds = {row["seed"] for row in confirmation}
    if source_seeds & primary_seeds or source_seeds & confirmation_seeds or primary_seeds & confirmation_seeds:
        raise ValueError("source, primary, and confirmation rollout seeds must be disjoint")
    if set(config["training_source_pool"]["init_state_indices"]) & set(confirm_inits):
        raise ValueError("training and confirmation initial states must be disjoint")
    return {
        "source_pool": source,
        "primary_evaluation": primary_plans,
        "confirmation_evaluation": confirmation,
        "baseline_and_source": source + primary_plans,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.campaign.read_text())
    plans = build_plans(config)
    for name, rows in plans.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
    summary = {name: len(rows) for name, rows in plans.items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

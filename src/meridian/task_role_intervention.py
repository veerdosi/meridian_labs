"""Build and validate the fixed evaluation plan for the task-role intervention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_evaluation_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    evaluation = config["evaluation"]
    plans = []
    seed = int(evaluation["seed"])
    for task_offset, task in enumerate(config["tasks"]):
        for state_offset, init_state in enumerate(evaluation["target_states"]):
            for repeat in range(int(evaluation["target_repeats_per_state"])):
                plans.append(
                    {
                        "id": f"target-t{task['task_id']}-i{init_state}-r{repeat}",
                        "task_suite": task["suite"],
                        "task_id": int(task["task_id"]),
                        "seed": seed + task_offset * 1000 + state_offset * 10 + repeat,
                        "init_state_index": int(init_state),
                        "replan_steps": int(evaluation["replan_steps"]),
                        "wait_steps": int(evaluation["wait_steps"]),
                        "max_steps": int(evaluation["max_steps"]),
                        "phase": "intervention_evaluation",
                        "evaluation_suite": "target",
                        "repeat": repeat,
                    }
                )
    for task_offset, task in enumerate(evaluation["regression_tasks"]):
        for state_offset, init_state in enumerate(evaluation["regression_states"]):
            plans.append(
                {
                    "id": f"regression-{task['suite']}-t{task['task_id']}-i{init_state}",
                    "task_suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "seed": seed + 10000 + task_offset * 100 + state_offset,
                    "init_state_index": int(init_state),
                    "replan_steps": int(evaluation["replan_steps"]),
                    "wait_steps": int(evaluation["wait_steps"]),
                    "max_steps": int(evaluation["max_steps"]),
                    "phase": "intervention_evaluation",
                    "evaluation_suite": "regression",
                }
            )
    target_count = sum(item["evaluation_suite"] == "target" for item in plans)
    regression_count = len(plans) - target_count
    if target_count != int(evaluation["target_trials"]):
        raise ValueError(f"target plan has {target_count}/{evaluation['target_trials']} trials")
    if regression_count != int(evaluation["regression_trials"]):
        raise ValueError(
            f"regression plan has {regression_count}/{evaluation['regression_trials']} trials"
        )
    if len({item["id"] for item in plans}) != len(plans):
        raise ValueError("evaluation plan contains duplicate IDs")
    return plans

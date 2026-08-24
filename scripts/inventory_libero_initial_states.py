#!/usr/bin/env python3
"""Inventory physical LIBERO initial states without policy inference or outcome rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from meridian.physical_boundary import validate_protocol_config
from meridian.rollout_integrity import (
    evaluate_goal_predicates,
    free_joint_positions,
    goal_metadata,
    reserve_results_path,
    simulator_state_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_protocol_config(config)
    reserve_results_path(args.output)
    suites = {name: benchmark.get_benchmark_dict()[name]() for name in {task["suite"] for task in config["task_set"]}}
    with args.output.open("a") as stream:
        for task_spec in config["task_set"]:
            suite = suites[task_spec["suite"]]
            task = suite.get_task(int(task_spec["task_id"]))
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            try:
                env.reset()
                states = suite.get_task_init_states(int(task_spec["task_id"]))
                for index, state in enumerate(states):
                    env.set_init_state(state)
                    goals = goal_metadata(env)
                    initial_predicates = evaluate_goal_predicates(env, goals["predicates"])
                    record = {
                        "task_suite": task_spec["suite"],
                        "task_id": int(task_spec["task_id"]),
                        "init_state_index": index,
                        "initial_sim_state_sha256": simulator_state_sha256(env.sim.data.qpos, env.sim.data.qvel),
                        "initial_free_joint_positions": free_joint_positions(env.sim),
                        "initial_sim_qpos": [float(value) for value in env.sim.data.qpos],
                        "goal_predicates": goals["predicates"],
                        "goal_already_satisfied": bool(initial_predicates.all()),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            finally:
                env.close()


if __name__ == "__main__":
    main()

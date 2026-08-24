#!/usr/bin/env python3
"""Parameterized LIBERO client for a separately served OpenPI policy."""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from meridian.recording import write_evidence_videos
from meridian.rollout_integrity import (
    canonical_sha256,
    contact_pairs,
    evaluate_goal_predicates,
    file_sha256,
    free_joint_positions,
    goal_argument_positions,
    goal_metadata,
    initial_physical_features,
    pad_contact_pairs,
    reserve_results_path,
    simulator_metadata,
    simulator_state_sha256,
    validate_plans,
)

DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0])
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def state_vector(obs: dict) -> np.ndarray:
    quat = np.array(obs["robot0_eef_quat"], copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    axis_angle = (
        np.zeros(3)
        if math.isclose(denominator, 0.0)
        else quat[:3] * 2.0 * math.acos(quat[3]) / denominator
    )
    return np.concatenate((obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]))


def environment(task, seed: int) -> OffScreenRenderEnv:
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env


def run_plan(client: WebsocketClientPolicy, suite, plan: dict, output: Path) -> dict:
    started = time.monotonic()
    task_id, seed = int(plan["task_id"]), int(plan["seed"])
    task = suite.get_task(task_id)
    env = environment(task, seed)
    episode_id = plan.get("id", f"task{task_id}-seed{seed}")
    episode_dir = output / "episodes" / str(episode_id)
    episode_dir.mkdir(parents=True, exist_ok=True)
    clean_images, policy_images, wrist_images, states, actions = [], [], [], [], []
    sim_qpos, sim_qvel, contact_counts, sim_contact_pairs = [], [], [], []
    goal_predicates_before, goal_predicates_after, goal_positions_after = [], [], []
    done, failure_phase, inference_seconds = False, "timeout", 0.0
    try:
        env.reset()
        initial_states = suite.get_task_init_states(task_id)
        init_index = int(plan.get("init_state_index", seed % len(initial_states)))
        obs = env.set_init_state(initial_states[init_index])
        obs = env.regenerate_obs_from_state(env.get_sim_state())
        initial_state_hash = simulator_state_sha256(env.sim.data.qpos, env.sim.data.qvel)
        initial_objects = free_joint_positions(env.sim)
        physical_features = initial_physical_features(env.sim)
        initial_qpos = np.array(env.sim.data.qpos, copy=True)
        initial_qvel = np.array(env.sim.data.qvel, copy=True)
        sim_schema = simulator_metadata(env.sim)
        goal_schema = goal_metadata(env)
        sim_schema["goal_predicates"] = goal_schema["predicates"]
        sim_schema["goal_arguments"] = goal_schema["arguments"]
        action_plan: collections.deque = collections.deque()
        wait_steps = int(plan.get("wait_steps", 10))
        replan_steps = int(plan.get("replan_steps", 5))
        max_steps = int(plan.get("max_steps", MAX_STEPS[plan["task_suite"]])) + wait_steps
        for step in range(max_steps):
            if step < wait_steps:
                obs, _, done, _ = env.step(DUMMY_ACTION.tolist())
                if done:
                    raise ValueError(
                        f"plan {episode_id} reaches the goal during settling; invalid initial state"
                    )
                continue
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            policy_image = image
            clean_image = image.copy()
            if not action_plan:
                element = {
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(policy_image, 224, 224)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, 224, 224)
                    ),
                    "observation/state": state_vector(obs),
                    "prompt": str(task.language),
                }
                inference_started = time.monotonic()
                chunk = client.infer(element)["actions"]
                inference_seconds += time.monotonic() - inference_started
                action_plan.extend(chunk[:replan_steps])
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            clean_images.append(clean_image)
            policy_images.append(policy_image)
            wrist_images.append(wrist)
            states.append(state_vector(obs))
            actions.append(action)
            sim_qpos.append(np.array(env.sim.data.qpos, copy=True))
            sim_qvel.append(np.array(env.sim.data.qvel, copy=True))
            contact_counts.append(int(env.sim.data.ncon))
            sim_contact_pairs.append(contact_pairs(env.sim))
            goal_predicates_before.append(
                evaluate_goal_predicates(env, goal_schema["predicates"])
            )
            obs, _, done, _ = env.step(action.tolist())
            goal_predicates_after.append(
                evaluate_goal_predicates(env, goal_schema["predicates"])
            )
            goal_positions_after.append(
                goal_argument_positions(env, goal_schema["arguments"])
            )
            if done:
                failure_phase = "complete"
                break
        trace_path = episode_dir / "trajectory.npz"
        np.savez_compressed(
            trace_path,
            image=policy_images,
            clean_observer_image=clean_images,
            policy_image=policy_images,
            wrist_image=wrist_images,
            state=states,
            actions=actions,
            sim_qpos=sim_qpos,
            sim_qvel=sim_qvel,
            contact_count=contact_counts,
            contact_geom_ids=pad_contact_pairs(sim_contact_pairs),
            initial_sim_qpos=initial_qpos,
            initial_sim_qvel=initial_qvel,
            goal_predicate_satisfied_before=goal_predicates_before,
            goal_predicate_satisfied_after=goal_predicates_after,
            goal_argument_positions_after=goal_positions_after,
        )
        videos = (
            write_evidence_videos(
                episode_dir,
                clean_images,
                policy_images,
                wrist_images,
                parameters=plan,
                success=bool(done),
            )
            if policy_images
            else {}
        )
        prompt = str(task.language)
        return {
            "id": episode_id,
            "task_suite": plan["task_suite"],
            "task_id": task_id,
            "task": prompt,
            "prompt": prompt,
            "seed": seed,
            "init_state_index": init_index,
            "parameters": {
                key: value
                for key, value in plan.items()
                if key not in {"id", "task_suite", "task_id", "seed"}
            },
            "success": bool(done),
            "score": float(done),
            "phase_labels": [failure_phase],
            "steps": len(actions),
            "duration_seconds": time.monotonic() - started,
            "inference_seconds": inference_seconds,
            "trace": str(trace_path),
            "trace_sha256": file_sha256(trace_path),
            "plan_sha256": canonical_sha256(plan),
            "initial_sim_state_sha256": initial_state_hash,
            "initial_free_joint_positions": initial_objects,
            "initial_physical_features": physical_features,
            "initial_sim_qpos": [float(value) for value in initial_qpos],
            "simulator_schema": sim_schema,
            "video": videos.get("diagnostic"),
            "videos": videos,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    plans = validate_plans(
        [json.loads(line) for line in args.plans.read_text().splitlines() if line.strip()]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    client = WebsocketClientPolicy(args.host, args.port)
    suites = {
        name: benchmark.get_benchmark_dict()[name]()
        for name in {plan["task_suite"] for plan in plans}
    }
    results_path = args.output / "rollouts.jsonl"
    reserve_results_path(results_path)
    with results_path.open("a") as stream:
        for plan in plans:
            result = run_plan(client, suites[plan["task_suite"]], plan, args.output)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

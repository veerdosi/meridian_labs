#!/usr/bin/env python3
"""Generate verified contrastive LIBERO demonstrations with a privileged OSC expert."""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from libero.libero import benchmark
from libero.libero.envs import SegmentationRenderEnv
from robosuite.utils import transform_utils as T
from robosuite.utils.control_utils import orientation_error

from meridian.recording import write_evidence_videos
from meridian.rollout_integrity import (
    contact_pairs,
    evaluate_goal_predicates,
    file_sha256,
    goal_argument_positions,
    goal_metadata,
    pad_contact_pairs,
    reserve_results_path,
    simulator_metadata,
)
from meridian.task_role_repair import validate_repair_config

OPEN_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def state_vector(obs: dict[str, Any]) -> np.ndarray:
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    clipped_w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - clipped_w**2))
    axis_angle = (
        np.zeros(3)
        if math.isclose(denominator, 0.0)
        else quat[:3] * 2.0 * math.acos(clipped_w) / denominator
    )
    return np.concatenate(
        (obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"])
    ).astype(np.float32)


def task_config(config: dict, task_id: int) -> dict:
    matches = [task for task in config["tasks"] if int(task["task_id"]) == task_id]
    if len(matches) != 1:
        raise ValueError(f"task {task_id} is not uniquely configured")
    return matches[0]


def make_environment(bddl: Path, seed: int) -> SegmentationRenderEnv:
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=128,
        camera_widths=128,
        camera_segmentations="instance",
        horizon=1000,
        ignore_done=True,
    )
    env.seed(seed)
    return env


def object_position(env: Any, name: str) -> np.ndarray:
    return np.asarray(env.env.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)


def set_object_pose(env: Any, joint: str, name: str, placement: dict, profile: dict) -> None:
    desired_xy = np.asarray(placement["xy_m"], dtype=np.float64)
    surface = str(placement["surface"])
    z_key = f"{surface}_geom_z_m"
    if z_key not in profile:
        raise ValueError(f"{name} has no validated height for surface {surface}")
    desired_geom = np.asarray([desired_xy[0], desired_xy[1], float(profile[z_key])])
    current_geom = object_position(env, name)
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    if qpos.shape != (7,):
        raise ValueError(f"{joint} is not a free joint")
    qpos[:3] += desired_geom - current_geom
    env.sim.data.set_joint_qpos(joint, qpos)
    joint_id = env.sim.model.joint_name2id(joint)
    dof_address = int(env.sim.model.jnt_dofadr[joint_id])
    env.sim.data.qvel[dof_address : dof_address + 6] = 0.0
    env.sim.forward()


def visible_pixels(env: Any, obs: dict[str, Any], object_names: list[str]) -> dict[str, int]:
    key = "agentview_segmentation_instance"
    if key not in obs:
        raise ValueError(f"segmentation observation lacks {key}")
    segmentation = np.asarray(obs[key]).squeeze(-1)
    counts = {}
    for name in object_names:
        if name not in env.instance_to_id:
            raise ValueError(
                f"segmentation mapping lacks object {name}: {sorted(env.instance_to_id)}"
            )
        counts[name] = int(np.count_nonzero(segmentation == int(env.instance_to_id[name])))
    return counts


class EpisodeRecorder:
    def __init__(self, env: Any, goal_schema: dict, commanded: str, other: str, maximum: int):
        self.env = env
        self.goal_schema = goal_schema
        self.commanded = commanded
        self.other = other
        self.maximum = maximum
        self.data: dict[str, list] = collections.defaultdict(list)

    def step(self, obs: dict[str, Any], action: np.ndarray, stage: str) -> tuple[dict, bool]:
        if len(self.data["actions"]) >= self.maximum:
            raise RuntimeError("expert exceeded the locked maximum step count")
        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        self.data["clean_observer_image"].append(image)
        self.data["policy_image"].append(image.copy())
        self.data["wrist_image"].append(wrist)
        self.data["state"].append(state_vector(obs))
        self.data["actions"].append(np.asarray(action, dtype=np.float32))
        self.data["sim_qpos"].append(np.asarray(self.env.sim.data.qpos, dtype=np.float64).copy())
        self.data["sim_qvel"].append(np.asarray(self.env.sim.data.qvel, dtype=np.float64).copy())
        self.data["contact_count"].append(int(self.env.sim.data.ncon))
        self.data["contact_pairs"].append(contact_pairs(self.env.sim))
        self.data["goal_before"].append(
            evaluate_goal_predicates(self.env, self.goal_schema["predicates"])
        )
        self.data["stage"].append(stage)
        next_obs, _, _, _ = self.env.step(np.asarray(action, dtype=np.float32).tolist())
        goal_after = evaluate_goal_predicates(self.env, self.goal_schema["predicates"])
        self.data["goal_after"].append(goal_after)
        self.data["goal_positions"].append(
            goal_argument_positions(self.env, self.goal_schema["arguments"])
        )
        self.data["commanded_position"].append(object_position(self.env, self.commanded))
        self.data["other_position"].append(object_position(self.env, self.other))
        return next_obs, bool(np.all(goal_after))


def osc_action(
    obs: dict[str, Any], desired_position: np.ndarray, desired_axis_angle: np.ndarray, gripper: float
) -> tuple[np.ndarray, float, float]:
    position_error = np.asarray(desired_position) - np.asarray(obs["robot0_eef_pos"])
    desired_matrix = T.quat2mat(T.axisangle2quat(np.asarray(desired_axis_angle)))
    current_matrix = T.quat2mat(np.asarray(obs["robot0_eef_quat"]))
    rotation_error = orientation_error(desired_matrix, current_matrix)
    action = np.concatenate(
        (
            np.clip(position_error / 0.05, -1.0, 1.0),
            np.clip(rotation_error / 0.5, -1.0, 1.0),
            [float(gripper)],
        )
    ).astype(np.float32)
    return action, float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))


def move_to(
    recorder: EpisodeRecorder,
    obs: dict[str, Any],
    position: np.ndarray,
    axis_angle: np.ndarray,
    gripper: float,
    stage: str,
    maximum_steps: int,
) -> tuple[dict, bool, bool]:
    stable = 0
    goal = False
    for _ in range(maximum_steps):
        action, position_error, rotation_error = osc_action(obs, position, axis_angle, gripper)
        obs, goal = recorder.step(obs, action, stage)
        stable = stable + 1 if position_error < 0.008 and rotation_error < 0.08 else 0
        if stable >= 3:
            return obs, goal, True
    return obs, goal, False


def hold(
    recorder: EpisodeRecorder,
    obs: dict[str, Any],
    position: np.ndarray,
    axis_angle: np.ndarray,
    gripper: float,
    stage: str,
    steps: int,
) -> tuple[dict, bool]:
    goal = False
    for _ in range(steps):
        action, _, _ = osc_action(obs, position, axis_angle, gripper)
        obs, goal = recorder.step(obs, action, stage)
    return obs, goal


def write_trace(path: Path, recorder: EpisodeRecorder) -> None:
    data = recorder.data
    np.savez_compressed(
        path,
        image=np.asarray(data["policy_image"], dtype=np.uint8),
        clean_observer_image=np.asarray(data["clean_observer_image"], dtype=np.uint8),
        policy_image=np.asarray(data["policy_image"], dtype=np.uint8),
        wrist_image=np.asarray(data["wrist_image"], dtype=np.uint8),
        state=np.asarray(data["state"], dtype=np.float32),
        actions=np.asarray(data["actions"], dtype=np.float32),
        sim_qpos=np.asarray(data["sim_qpos"], dtype=np.float64),
        sim_qvel=np.asarray(data["sim_qvel"], dtype=np.float64),
        contact_count=np.asarray(data["contact_count"], dtype=np.int32),
        contact_geom_ids=pad_contact_pairs(data["contact_pairs"]),
        goal_predicate_satisfied_before=np.asarray(data["goal_before"], dtype=bool),
        goal_predicate_satisfied_after=np.asarray(data["goal_after"], dtype=bool),
        goal_argument_positions_after=np.asarray(data["goal_positions"], dtype=np.float64),
        commanded_object_position_after=np.asarray(data["commanded_position"], dtype=np.float64),
        other_object_position_after=np.asarray(data["other_position"], dtype=np.float64),
        stage=np.asarray(data["stage"], dtype="U32"),
    )


def run_plan(
    plan: dict,
    config: dict,
    suite: Any,
    project_root: Path,
    output: Path,
    write_videos: bool,
) -> dict:
    started = time.monotonic()
    task = task_config(config, int(plan["task_id"]))
    bddl = project_root / str(plan["bddl"])
    env = make_environment(bddl, int(plan["seed"]))
    episode_dir = output / "episodes" / str(plan["id"])
    episode_dir.mkdir(parents=True, exist_ok=False)
    try:
        env.reset()
        states = suite.get_task_init_states(int(plan["task_id"]))
        obs = env.set_init_state(states[int(plan["init_state_index"])])
        profile_by_object = task["expert_profiles"]
        joint_by_object = {
            str(variant["commanded_object"]): str(variant["commanded_joint"])
            for variant in task["role_variants"]
        }
        for name, placement in plan["layout"]["objects"].items():
            set_object_pose(
                env,
                joint_by_object[str(name)],
                str(name),
                placement,
                profile_by_object[str(name)],
            )
        obs = env.regenerate_obs_from_state(env.get_sim_state())
        for _ in range(20):
            obs, _, _, _ = env.step(OPEN_ACTION.tolist())
        commanded = str(plan["commanded_object"])
        other = str(plan["other_object"])
        initial_commanded = object_position(env, commanded)
        initial_other = object_position(env, other)
        initial_pixels = visible_pixels(env, obs, [commanded, other])
        minimum_pixels = int(config["expert_acceptance"]["minimum_initial_visible_pixels_per_object"])
        if any(value < minimum_pixels for value in initial_pixels.values()):
            raise ValueError(f"object visibility below threshold: {initial_pixels}")
        goal_schema = goal_metadata(env)
        if goal_schema["predicates"] != [list(plan["goal_predicate"])]:
            raise ValueError("custom BDDL goal does not match the locked role variant")
        if bool(np.all(evaluate_goal_predicates(env, goal_schema["predicates"]))):
            raise ValueError("generated initial state already satisfies the goal")

        maximum = int(config["expert_acceptance"]["maximum_steps"])
        recorder = EpisodeRecorder(env, goal_schema, commanded, other, maximum)
        profile = profile_by_object[commanded]
        axis_angle = np.asarray(profile["grasp_axis_angle_rad"], dtype=np.float64)
        grasp = initial_commanded + np.asarray(profile["grasp_offset_world_m"], dtype=np.float64)
        destination = object_position(env, str(plan["destination"]))
        place = destination + np.asarray(profile["place_offset_world_m"], dtype=np.float64)
        stages = [
            ("pregrasp", grasp + [0.0, 0.0, 0.14], -1.0, 70),
            ("grasp", grasp, -1.0, 45),
        ]
        reached = True
        for stage, position, gripper, limit in stages:
            obs, _goal, reached = move_to(
                recorder, obs, np.asarray(position), axis_angle, gripper, stage, limit
            )
            if not reached:
                break
        failure = None if reached else f"failed_to_reach_{stage}"
        if reached:
            obs, _goal = hold(recorder, obs, grasp, axis_angle, 1.0, "close", 18)
            obs, _goal, reached = move_to(
                recorder, obs, grasp + [0.0, 0.0, 0.16], axis_angle, 1.0, "lift", 55
            )
            failure = None if reached else "failed_to_reach_lift"
        if reached:
            obs, _goal, reached = move_to(
                recorder, obs, place + [0.0, 0.0, 0.14], axis_angle, 1.0, "preplace", 70
            )
            failure = None if reached else "failed_to_reach_preplace"
        if reached:
            obs, _goal, reached = move_to(
                recorder, obs, place, axis_angle, 1.0, "place", 55
            )
            failure = None if reached else "failed_to_reach_place"
        if reached:
            obs, _goal = hold(recorder, obs, place, axis_angle, -1.0, "release", 20)
            obs, _goal, reached = move_to(
                recorder, obs, place + [0.0, 0.0, 0.14], axis_angle, -1.0, "retreat", 45
            )
            failure = None if reached else "failed_to_reach_retreat"
        if reached:
            obs, _goal = hold(
                recorder, obs, place + [0.0, 0.0, 0.14], axis_angle, -1.0, "settle", 20
            )

        trace = episode_dir / "trajectory.npz"
        write_trace(trace, recorder)
        commanded_positions = np.asarray(recorder.data["commanded_position"])
        other_positions = np.asarray(recorder.data["other_position"])
        target_motion = float(np.max(np.linalg.norm(commanded_positions - initial_commanded, axis=1)))
        other_motion = float(np.max(np.linalg.norm(other_positions - initial_other, axis=1)))
        final_goal = bool(np.all(evaluate_goal_predicates(env, goal_schema["predicates"])))
        acceptance = config["expert_acceptance"]
        checks = {
            "goal": final_goal,
            "target_motion": target_motion >= float(acceptance["minimum_target_translation_m"]),
            "other_stationary": other_motion
            <= float(acceptance["maximum_final_distractor_translation_m"]),
            "finite": all(
                np.isfinite(np.asarray(recorder.data[name])).all()
                for name in ("state", "actions", "sim_qpos", "sim_qvel")
            ),
            "controller_completed": reached,
        }
        accepted = all(checks.values())
        videos = {}
        if write_videos:
            videos = write_evidence_videos(
                episode_dir,
                recorder.data["clean_observer_image"],
                recorder.data["policy_image"],
                recorder.data["wrist_image"],
                parameters=plan,
                success=accepted,
            )
        return {
            "schema": "task-role-expert-episode-v1",
            "id": plan["id"],
            "arm": plan["arm"],
            "task_suite": plan["task_suite"],
            "task_id": int(plan["task_id"]),
            "role_variant": plan["role_variant"],
            "prompt": plan["prompt"],
            "commanded_object": commanded,
            "other_object": other,
            "goal_predicate": plan["goal_predicate"],
            "layout": plan["layout"],
            "init_state_index": int(plan["init_state_index"]),
            "state_spec_sha256": plan["state_spec_sha256"],
            "accepted": accepted,
            "checks": checks,
            "failure": failure,
            "steps": len(recorder.data["actions"]),
            "target_translation_m": target_motion,
            "other_translation_m": other_motion,
            "initial_visible_pixels": initial_pixels,
            "trace": str(trace),
            "trace_sha256": file_sha256(trace),
            "videos": videos,
            "simulator_schema": simulator_metadata(env.sim),
            "duration_seconds": time.monotonic() - started,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-videos", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_repair_config(config, args.project_root)
    plans = [json.loads(line) for line in args.plans.read_text().splitlines() if line.strip()]
    if not plans:
        raise ValueError("expert plan is empty")
    args.output.mkdir(parents=True, exist_ok=False)
    results_path = args.output / "episodes.jsonl"
    reserve_results_path(results_path)
    suites = {
        name: benchmark.get_benchmark_dict()[name]() for name in {plan["task_suite"] for plan in plans}
    }
    with results_path.open("a") as stream:
        for plan in plans:
            try:
                result = run_plan(
                    plan,
                    config,
                    suites[plan["task_suite"]],
                    args.project_root,
                    args.output,
                    args.write_videos,
                )
            # One invalid candidate must be recorded and rejected without erasing later attempts.
            except Exception as error:  # noqa: BLE001
                result = {
                    "schema": "task-role-expert-episode-v1",
                    "id": plan.get("id"),
                    "arm": plan.get("arm"),
                    "task_id": plan.get("task_id"),
                    "role_variant": plan.get("role_variant"),
                    "accepted": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=8),
                }
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

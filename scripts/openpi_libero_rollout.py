#!/usr/bin/env python3
"""Parameterized LIBERO client for a separately served OpenPI policy."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0])
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def apply_sim_parameters(env: OffScreenRenderEnv, parameters: dict[str, float | str]) -> None:
    camera_id = env.sim.model.camera_name2id("agentview")
    env.sim.model.cam_pos[camera_id, 0] += float(parameters.get("camera_x", 0.0))
    yaw = math.radians(float(parameters.get("camera_yaw_deg", 0.0)))
    yaw_quat = np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    env.sim.model.cam_quat[camera_id] = quat_multiply(yaw_quat, env.sim.model.cam_quat[camera_id])
    joint_name = parameters.get("object_joint")
    if joint_name:
        joint_id = env.sim.model.joint_name2id(str(joint_name))
        qpos_address = env.sim.model.jnt_qposadr[joint_id]
        env.sim.data.qpos[qpos_address] += float(parameters.get("object_x", 0.0))
        env.sim.data.qpos[qpos_address + 1] += float(parameters.get("object_y", 0.0))
    env.sim.forward()


def perturb_image(
    image: np.ndarray, parameters: dict[str, float | str], rng: np.random.Generator
) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32) * float(parameters.get("brightness", 1.0))
    result = np.clip(result, 0, 255).astype(np.uint8)
    occlusion = float(parameters.get("occlusion", 0.0))
    if occlusion > 0:
        height, width = result.shape[:2]
        side = int(math.sqrt(min(0.8, occlusion)) * min(height, width))
        x = int(rng.integers(0, max(1, width - side)))
        y = int(rng.integers(0, max(1, height - side)))
        result[y : y + side, x : x + side] = 0
    for _ in range(int(parameters.get("visual_distractors", 0))):
        height, width = result.shape[:2]
        side = max(4, min(height, width) // 18)
        x = int(rng.integers(0, max(1, width - side)))
        y = int(rng.integers(0, max(1, height - side)))
        result[y : y + side, x : x + side] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return result


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
    rng = np.random.default_rng(seed)
    episode_id = plan.get("id", f"task{task_id}-seed{seed}")
    episode_dir = output / "episodes" / str(episode_id)
    episode_dir.mkdir(parents=True, exist_ok=True)
    observations, wrist_images, states, actions, replay = [], [], [], [], []
    done, failure_phase, inference_seconds = False, "timeout", 0.0
    try:
        env.reset()
        initial_states = suite.get_task_init_states(task_id)
        init_index = int(plan.get("init_state_index", seed % len(initial_states)))
        obs = env.set_init_state(initial_states[init_index])
        apply_sim_parameters(env, plan)
        obs = env._get_observations()
        action_plan: collections.deque = collections.deque()
        wait_steps = int(plan.get("wait_steps", 10))
        replan_steps = int(plan.get("replan_steps", 5))
        max_steps = int(plan.get("max_steps", MAX_STEPS[plan["task_suite"]])) + wait_steps
        for step in range(max_steps):
            if step < wait_steps:
                obs, _, done, _ = env.step(DUMMY_ACTION.tolist())
                continue
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            policy_image = perturb_image(image, plan, rng)
            replay.append(policy_image)
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
            noise = float(plan.get("action_noise", 0.0))
            if noise:
                action[:6] += rng.normal(0.0, noise, 6)
            observations.append(image)
            wrist_images.append(wrist)
            states.append(state_vector(obs))
            actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                failure_phase = "complete"
                break
        trace_path = episode_dir / "trajectory.npz"
        np.savez_compressed(
            trace_path, image=observations, wrist_image=wrist_images, state=states, actions=actions
        )
        video_path = episode_dir / "rollout.mp4"
        if replay:
            iio.imwrite(video_path, np.asarray(replay), fps=10)
        trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        return {
            "id": episode_id,
            "task_suite": plan["task_suite"],
            "task_id": task_id,
            "task": str(task.language),
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
            "trace_sha256": trace_hash,
            "video": str(video_path) if replay else None,
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
    plans = [json.loads(line) for line in args.plans.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    client = WebsocketClientPolicy(args.host, args.port)
    suites = {
        name: benchmark.get_benchmark_dict()[name]()
        for name in {plan["task_suite"] for plan in plans}
    }
    results_path = args.output / "rollouts.jsonl"
    with results_path.open("a") as stream:
        for plan in plans:
            result = run_plan(client, suites[plan["task_suite"]], plan, args.output)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

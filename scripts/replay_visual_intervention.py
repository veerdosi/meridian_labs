#!/usr/bin/env python3
"""Replay successful LIBERO action traces under targeted visual parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from libero.libero import benchmark

from scripts.openpi_libero_rollout import (
    DUMMY_ACTION,
    apply_sim_parameters,
    camera_pose,
    environment,
    perturb_image,
    render_at_camera_pose,
    state_vector,
    write_evidence_videos,
)
from scripts.replay_source_contract import load_replay_sources


def replay(base: dict, actions: np.ndarray, plan: dict, output: Path) -> dict:
    started = time.monotonic()
    suite = benchmark.get_benchmark_dict()[base["task_suite"]]()
    task = suite.get_task(base["task_id"])
    env = environment(task, base["seed"])
    episode_id = plan["id"]
    episode_dir = output / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(plan["seed"]))
    clean_images, policy_images, wrist_images, states, replayed_actions = [], [], [], [], []
    done = False
    try:
        env.reset()
        initial_states = suite.get_task_init_states(base["task_id"])
        obs = env.set_init_state(initial_states[base["init_state_index"]])
        observer_pose = camera_pose(env)
        apply_sim_parameters(env, plan)
        obs = env.regenerate_obs_from_state(env.get_sim_state())
        for _ in range(int(base["parameters"].get("wait_steps", 10))):
            obs, _, _, _ = env.step(DUMMY_ACTION.tolist())
        for action in actions:
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            policy_image = perturb_image(image, plan, rng)
            clean_image = render_at_camera_pose(env, observer_pose)
            clean_images.append(clean_image)
            policy_images.append(policy_image)
            wrist_images.append(wrist)
            states.append(state_vector(obs))
            replayed_actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
        trace_path = episode_dir / "trajectory.npz"
        np.savez_compressed(
            trace_path,
            image=policy_images,
            clean_observer_image=clean_images,
            policy_image=policy_images,
            wrist_image=wrist_images,
            state=states,
            actions=replayed_actions,
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
        return {
            "id": episode_id,
            "task_suite": base["task_suite"],
            "task_id": base["task_id"],
            "task": base["task"],
            "seed": int(plan["seed"]),
            "init_state_index": base["init_state_index"],
            "parameters": {
                key: value
                for key, value in plan.items()
                if key not in {"id", "task_suite", "task_id", "seed", "source_rollout_id"}
            },
            "success": bool(done),
            "score": float(done),
            "phase_labels": ["complete" if done else "replay_failed"],
            "steps": len(replayed_actions),
            "duration_seconds": time.monotonic() - started,
            "inference_seconds": 0.0,
            "trace": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "video": videos.get("diagnostic"),
            "videos": videos,
            "provenance": {
                "kind": "successful_action_replay",
                "source_rollout_id": base["id"],
                "source_trace_sha256": base["trace_sha256"],
                "physics_seed": base["seed"],
                "visual_seed": int(plan["seed"]),
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rollout", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path)
    args = parser.parse_args()
    plans = [json.loads(line) for line in args.plans.read_text().splitlines() if line]
    source_ids = {plan["source_rollout_id"] for plan in plans if plan.get("source_rollout_id")}
    sources = load_replay_sources(
        args.base_rollout,
        required_ids=source_ids or None,
        trace_root=args.trace_root,
    )
    base_records = list(sources.values())
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "rollouts.jsonl"
    with output_path.open("w") as stream:
        for index, plan in enumerate(plans):
            source_id = plan.get("source_rollout_id")
            if source_id is None:
                base = base_records[index % len(base_records)]
            else:
                try:
                    base = sources[source_id]
                except KeyError as error:
                    raise ValueError(f"unknown source_rollout_id: {source_id}") from error
            with np.load(base["trace"]) as source:
                actions = np.array(source["actions"], copy=True)
            record = replay(base, actions, plan, args.output)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert verified simulator-expert episodes plus fixed replay into one LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expert(episode: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trace = Path(str(episode["trace"]))
    if file_sha256(trace) != episode["trace_sha256"]:
        raise ValueError(f"expert trace hash mismatch: {trace}")
    with np.load(trace) as trajectory:
        image = np.asarray(trajectory["policy_image"], dtype=np.uint8)
        wrist = np.asarray(trajectory["wrist_image"], dtype=np.uint8)
        state = np.asarray(trajectory["state"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        goal = np.asarray(trajectory["goal_predicate_satisfied_after"], dtype=bool)
    if not bool(np.all(goal[-1])):
        raise ValueError(f"expert episode does not end successful: {trace}")
    return image, wrist, state, actions


def load_replay(episode: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = Path(str(episode["source"]))
    if file_sha256(source) != episode["source_sha256"]:
        raise ValueError(f"replay source hash mismatch: {source}")
    demo = str(episode["demo"])
    with h5py.File(source, "r") as archive:
        group = archive[f"data/{demo}"]
        image = np.asarray(group["obs/agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(group["obs/eye_in_hand_rgb"], dtype=np.uint8)
        state = np.concatenate(
            (np.asarray(group["obs/ee_states"]), np.asarray(group["obs/gripper_states"])),
            axis=1,
        ).astype(np.float32)
        actions = np.asarray(group["actions"], dtype=np.float32)
        rewards = np.asarray(group["rewards"])
        dones = np.asarray(group["dones"])
    if not bool(rewards[-1]) or not bool(dones[-1]):
        raise ValueError(f"replay episode is not labeled successful: {source}:{demo}")
    # Official LIBERO HDF5 images use raw robosuite orientation. Match π0.5 inference.
    return (
        np.ascontiguousarray(image[:, ::-1, ::-1]),
        np.ascontiguousarray(wrist[:, ::-1, ::-1]),
        state,
        actions,
    )


def validate_arrays(
    identifier: str,
    image: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    actions: np.ndarray,
) -> None:
    lengths = {len(value) for value in (image, wrist, state, actions)}
    if len(lengths) != 1 or next(iter(lengths), 0) == 0:
        raise ValueError(f"unaligned or empty episode: {identifier}")
    if image.shape[1:] != (128, 128, 3) or wrist.shape[1:] != (128, 128, 3):
        raise ValueError(f"unexpected image shape: {identifier}")
    if state.shape[1:] != (8,) or actions.shape[1:] != (7,):
        raise ValueError(f"unexpected state/action shape: {identifier}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError(f"non-finite state/action: {identifier}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-new", type=int, required=True)
    parser.add_argument("--expected-replay", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    episodes = list(manifest["episodes"])
    observed_new = sum(item["kind"] == "expert_npz" for item in episodes)
    observed_replay = sum(item["kind"] == "official_hdf5" for item in episodes)
    if (observed_new, observed_replay) != (args.expected_new, args.expected_replay):
        raise ValueError(
            f"manifest has new/replay {observed_new}/{observed_replay}, expected "
            f"{args.expected_new}/{args.expected_replay}"
        )
    dataset_root = args.output_root / args.repo_id
    if dataset_root.exists():
        raise FileExistsError(f"refusing to reuse dataset root: {dataset_root}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )
    provenance = []
    for episode in episodes:
        identifier = str(episode["id"])
        if episode["kind"] == "expert_npz":
            image, wrist, state, actions = load_expert(episode)
        elif episode["kind"] == "official_hdf5":
            image, wrist, state, actions = load_replay(episode)
        else:
            raise ValueError(f"unknown episode kind: {episode['kind']}")
        validate_arrays(identifier, image, wrist, state, actions)
        for index in range(len(actions)):
            dataset.add_frame(
                {
                    "image": image[index],
                    "wrist_image": wrist[index],
                    "state": state[index],
                    "actions": actions[index],
                    "task": str(episode["prompt"]),
                }
            )
        dataset.save_episode()
        provenance.append(
            {
                "id": identifier,
                "kind": episode["kind"],
                "prompt": episode["prompt"],
                "frames": len(actions),
            }
        )
    result = {
        "schema": "task-role-repair-conversion-v1",
        "repo_id": args.repo_id,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "new_episodes": observed_new,
        "replay_episodes": observed_replay,
        "episodes": provenance,
        "image_transform": {
            "expert_npz": "already_matches_pi05_inference",
            "official_hdf5": "rotate_180_to_match_pi05_inference",
        },
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

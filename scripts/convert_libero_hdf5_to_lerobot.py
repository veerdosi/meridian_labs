#!/usr/bin/env python3
"""Convert selected official LIBERO HDF5 demonstrations to a local LeRobot dataset."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    episodes = manifest["episodes"]
    if len(episodes) != args.expected_episodes:
        raise ValueError(f"manifest has {len(episodes)}/{args.expected_episodes} episodes")
    dataset_root = args.output_root / args.repo_id
    if dataset_root.exists():
        raise FileExistsError(f"refusing to reuse existing dataset root: {dataset_root}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
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
    verified_sources: dict[str, str] = {}
    output_episodes = []
    for episode in episodes:
        source = Path(episode["source"])
        source_key = str(source)
        if source_key not in verified_sources:
            observed_sha = file_sha256(source)
            if observed_sha != episode["source_sha256"]:
                raise ValueError(f"source hash mismatch: {source}")
            verified_sources[source_key] = observed_sha
        demo = str(episode["demo"])
        with h5py.File(source, "r") as archive:
            group = archive[f"data/{demo}"]
            images = np.asarray(group["obs/agentview_rgb"])
            wrists = np.asarray(group["obs/eye_in_hand_rgb"])
            state = np.concatenate(
                (np.asarray(group["obs/ee_states"]), np.asarray(group["obs/gripper_states"])),
                axis=1,
            )
            actions = np.asarray(group["actions"])
            rewards = np.asarray(group["rewards"])
            dones = np.asarray(group["dones"])
        lengths = {len(value) for value in (images, wrists, state, actions, rewards, dones)}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"unaligned or empty demonstration: {source}:{demo}")
        if images.shape[1:] != (128, 128, 3) or wrists.shape[1:] != (128, 128, 3):
            raise ValueError(f"unexpected image shape in {source}:{demo}")
        if state.shape[1:] != (8,) or actions.shape[1:] != (7,):
            raise ValueError(f"unexpected state/action shape in {source}:{demo}")
        if not bool(rewards[-1]) or not bool(dones[-1]):
            raise ValueError(f"demonstration is not labeled successful: {source}:{demo}")
        # LIBERO HDF5 stores raw robosuite images. π0.5's LIBERO inference contract rotates
        # both camera images by 180 degrees before they reach the policy.
        images = np.ascontiguousarray(images[:, ::-1, ::-1])
        wrists = np.ascontiguousarray(wrists[:, ::-1, ::-1])
        for index in range(len(actions)):
            dataset.add_frame(
                {
                    "image": images[index],
                    "wrist_image": wrists[index],
                    "state": state[index].astype(np.float32),
                    "actions": actions[index].astype(np.float32),
                    "task": str(episode["task"]),
                }
            )
        dataset.save_episode()
        output_episodes.append(
            {
                "source": source_key,
                "source_sha256": verified_sources[source_key],
                "demo": demo,
                "task": episode["task"],
                "frames": len(actions),
            }
        )
    print(
        json.dumps(
            {
                "schema": "meridian-libero-hdf5-conversion-v1",
                "repo_id": args.repo_id,
                "episodes": output_episodes,
                "image_transform": "rotate_180_to_match_pi05_libero_inference",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Hydrate and verify a converted intervention dataset before any training job uses it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.root / args.repo_id
    dataset = LeRobotDataset(args.repo_id, root=dataset_root)
    if dataset.num_episodes != args.expected_episodes:
        raise ValueError(f"dataset has {dataset.num_episodes}/{args.expected_episodes} episodes")
    if dataset.num_frames < args.expected_episodes:
        raise ValueError("dataset contains fewer frames than episodes")
    sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    observed_tasks = set()
    for index in sample_indices:
        sample = dataset[index]
        state = np.asarray(sample["state"])
        actions = np.asarray(sample["actions"])
        image = np.asarray(sample["image"])
        wrist = np.asarray(sample["wrist_image"])
        if state.shape != (8,) or actions.shape != (7,):
            raise ValueError("hydrated state or action shape is invalid")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("hydrated state or action contains non-finite values")
        if image.shape not in {(3, 128, 128), (128, 128, 3)}:
            raise ValueError(f"unexpected hydrated observer image shape: {image.shape}")
        if wrist.shape not in {(3, 128, 128), (128, 128, 3)}:
            raise ValueError(f"unexpected hydrated wrist image shape: {wrist.shape}")
        observed_tasks.add(str(sample["task"]))
    result = {
        "schema": "meridian-lerobot-dataset-verification-v1",
        "repo_id": args.repo_id,
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "sample_indices": sample_indices,
        "sample_tasks": sorted(observed_tasks),
        "state_shape": [8],
        "action_shape": [7],
        "image_shape": [128, 128, 3],
        "finite": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Convert successful Meridian trajectory NPZ files into an OpenPI-compatible LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from meridian.dataset_integrity import validate_training_records
from meridian.rollout_integrity import aligned_trajectory_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    args = parser.parse_args()
    dataset_root = args.output_root / args.repo_id
    if dataset_root.exists():
        raise FileExistsError(f"refusing to reuse existing dataset root: {dataset_root}")
    records = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()]
    selected = validate_training_records(records, expected_episodes=args.expected_episodes)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )
    for record in selected:
        trace_path = Path(record["trace"])
        with np.load(trace_path) as trajectory:
            images, wrists, states, actions = aligned_trajectory_arrays(
                trajectory, ("image", "wrist_image", "state", "actions")
            )
            for index in range(len(actions)):
                dataset.add_frame(
                    {
                        "image": images[index],
                        "wrist_image": wrists[index],
                        "state": states[index].astype(np.float32),
                        "actions": actions[index].astype(np.float32),
                        "task": record["task"],
                    }
                )
        dataset.save_episode()
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "episodes": len(selected),
                "source_ids": [record["id"] for record in selected],
                "trace_sha256": [record["trace_sha256"] for record in selected],
                "source_rollouts": str(args.rollouts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

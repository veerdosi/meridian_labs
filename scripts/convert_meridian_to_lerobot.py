#!/usr/bin/env python3
"""Convert successful Meridian trajectory NPZ files into an OpenPI-compatible LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root / args.repo_id,
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
    records = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()]
    selected = [record for record in records if record["success"] or args.include_failures]
    for record in selected:
        trajectory = np.load(record["trace"])
        for image, wrist, state, action in zip(
            trajectory["image"],
            trajectory["wrist_image"],
            trajectory["state"],
            trajectory["actions"],
            strict=True,
        ):
            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist,
                    "state": state.astype(np.float32),
                    "actions": action.astype(np.float32),
                    "task": record["task"],
                }
            )
        dataset.save_episode()
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "episodes": len(selected),
                "source_rollouts": str(args.rollouts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

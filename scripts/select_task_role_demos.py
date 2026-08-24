#!/usr/bin/env python3
"""Select nested, balanced, partition-safe demonstrations for the role-binding intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from meridian.demo_selection import select_partitioned_maximin, select_partitioned_random
from meridian.rollout_integrity import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    sources = json.loads(args.sources.read_text())
    inventory = [
        json.loads(line) for line in args.inventory.read_text().splitlines() if line.strip()
    ]
    doses = sorted(int(value) for value in config["data_contract"]["doses"])
    task_specs = sources["tasks"]
    if len(task_specs) != len(config["tasks"]):
        raise ValueError("source registry does not match the locked task count")
    if any(dose % len(task_specs) for dose in doses):
        raise ValueError("every dose must divide evenly across selected tasks")
    maximum_per_task = max(doses) // len(task_specs)
    allowed = [int(value) for value in config["partitions"]["training_init_states"]]
    forbidden = [
        int(value)
        for name in (
            "discovery_init_states",
            "confirmation_init_states",
            "untouched_holdout_init_states",
        )
        for value in config["partitions"][name]
    ]
    selected_tasks = []
    for source_spec in task_specs:
        key = (str(source_spec["suite"]), int(source_spec["task_id"]))
        if key not in {
            (str(task["suite"]), int(task["task_id"])) for task in config["tasks"]
        }:
            raise ValueError(f"unlocked task in source registry: {key}")
        source = Path(source_spec["source"])
        observed_sha = file_sha256(source)
        if observed_sha != source_spec["source_sha256"]:
            raise ValueError(f"source hash mismatch: {source}")
        target_address = int(source_spec["target_qpos_address"])
        distractor_address = int(source_spec["distractor_qpos_address"])
        task_inventory = {
            int(record["init_state_index"]): [
                *record["initial_sim_qpos"][target_address : target_address + 2],
                *record["initial_sim_qpos"][distractor_address : distractor_address + 2],
            ]
            for record in inventory
            if (str(record["task_suite"]), int(record["task_id"])) == key
        }
        if len(task_inventory) != 50:
            raise ValueError(f"inventory for {key} has {len(task_inventory)}/50 states")
        demo_features = {}
        with h5py.File(source, "r") as archive:
            problem = json.loads(archive["data"].attrs["problem_info"])
            if problem["language_instruction"] != source_spec["task"]:
                raise ValueError(f"prompt mismatch in {source}")
            for demo in archive["data"]:
                state = np.asarray(archive[f"data/{demo}/states"][0])
                demo_features[str(demo)] = [
                    *state[1 + target_address : 1 + target_address + 2],
                    *state[1 + distractor_address : 1 + distractor_address + 2],
                ]
        selection = select_partitioned_maximin(
            demo_features,
            task_inventory,
            allowed_indices=allowed,
            forbidden_indices=forbidden,
            count=maximum_per_task,
        )
        random_selection = select_partitioned_random(
            demo_features,
            task_inventory,
            allowed_indices=allowed,
            forbidden_indices=forbidden,
            excluded_demos=[item["demo"] for item in selection],
            count=maximum_per_task,
            seed=args.random_seed + int(source_spec["task_id"]),
        )
        selected_tasks.append(
            {
                **{key: source_spec[key] for key in source_spec},
                "selected": selection,
                "random_selected": random_selection,
            }
        )
    selected_tasks.sort(key=lambda item: (item["suite"], int(item["task_id"])))
    registry = {
        "schema": "meridian-task-role-targeted-selection-v1",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "sources": str(args.sources),
        "sources_sha256": file_sha256(args.sources),
        "inventory": str(args.inventory),
        "inventory_sha256": file_sha256(args.inventory),
        "nested_doses": True,
        "random_seed": args.random_seed,
        "random_control": "same tasks and partition-safe pool; seeded rank without geometry score",
        "tasks": selected_tasks,
    }
    (args.output_dir / "targeted-selection.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True)
    )
    for dose in doses:
        per_task = dose // len(selected_tasks)
        episodes = [
            {
                "source": task["source"],
                "source_sha256": task["source_sha256"],
                "demo": selected["demo"],
                "task": task["task"],
                "suite": task["suite"],
                "task_id": task["task_id"],
                "nearest_init_state_index": selected["nearest_init_state_index"],
                "nearest_normalized_distance": selected["nearest_normalized_distance"],
            }
            for task in selected_tasks
            for selected in task["selected"][:per_task]
        ]
        manifest = {
            "schema": "meridian-libero-hdf5-episode-manifest-v1",
            "arm": "targeted",
            "dose": dose,
            "balanced_tasks": True,
            "episodes": episodes,
        }
        (args.output_dir / f"targeted-dose-{dose}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        random_episodes = [
            {
                "source": task["source"],
                "source_sha256": task["source_sha256"],
                "demo": selected["demo"],
                "task": task["task"],
                "suite": task["suite"],
                "task_id": task["task_id"],
                "nearest_init_state_index": selected["nearest_init_state_index"],
                "nearest_normalized_distance": selected["nearest_normalized_distance"],
            }
            for task in selected_tasks
            for selected in task["random_selected"][:per_task]
        ]
        random_manifest = {
            "schema": "meridian-libero-hdf5-episode-manifest-v1",
            "arm": "random",
            "dose": dose,
            "balanced_tasks": True,
            "selection": "seeded random rank within the same partition-safe task pools",
            "episodes": random_episodes,
        }
        (args.output_dir / f"random-dose-{dose}.json").write_text(
            json.dumps(random_manifest, indent=2, sort_keys=True)
        )
    print(json.dumps({"doses": doses, "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()

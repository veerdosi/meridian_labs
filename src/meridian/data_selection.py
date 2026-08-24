"""Deterministic selection of unique successful intervention trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _vector(record: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    if "initial_sim_qpos" in record:
        return np.asarray(record["initial_sim_qpos"], dtype=float)
    if not names:
        raise ValueError("selected-task source record is missing initial_sim_qpos")
    positions = record["initial_free_joint_positions"]
    if set(positions) != set(names):
        raise ValueError("selected-task source records have incompatible physical schemas")
    return np.asarray(
        [coordinate for name in names for coordinate in positions[name]], dtype=float
    )


def select_data_arms(
    selection: Mapping[str, Any],
    boundary_record: Mapping[str, Any],
    successful_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict:
    selected = selection.get("selected")
    if not selected:
        raise ValueError("no selected boundary")
    unique = {str(record["id"]): record for record in successful_records if record["success"]}
    records = list(unique.values())
    task_records = [
        record for record in records
        if record["task_suite"] == selected["task_suite"]
        and int(record["task_id"]) == int(selected["task_id"])
    ]
    maximum_dose = max(int(value) for value in config["training"]["doses"])
    required = max(maximum_dose, int(config["thresholds"]["minimum_unique_targeted_sources"]))
    if len(task_records) < required:
        raise ValueError(
            f"only {len(task_records)} unique successful selected-task sources; {required} required"
        )
    if len(records) < maximum_dose:
        raise ValueError("insufficient unique successful records for the random arm")
    names = (
        []
        if "initial_sim_qpos" in boundary_record
        else sorted(boundary_record["initial_free_joint_positions"])
    )
    task_values = np.asarray([_vector(record, names) for record in task_records])
    scale = np.std(np.vstack((task_values, _vector(boundary_record, names))), axis=0)
    scale[scale < 1e-9] = 1.0
    boundary = _vector(boundary_record, names)
    targeted = sorted(
        task_records,
        key=lambda record: (
            float(np.linalg.norm((_vector(record, names) - boundary) / scale)),
            str(record["id"]),
        ),
    )
    rng = np.random.default_rng(int(config["training"]["source_seed"]))
    original = [task_records[int(index)] for index in rng.permutation(len(task_records))]
    random = [records[int(index)] for index in rng.permutation(len(records))]
    ordered = {"targeted": targeted, "random": random, "original_distribution": original}
    doses = {}
    for dose in sorted(int(value) for value in config["training"]["doses"]):
        doses[str(dose)] = {
            arm: [record["id"] for record in arm_records[:dose]]
            for arm, arm_records in ordered.items()
        }
    return {
        "schema": "meridian-physical-data-selection-v1",
        "doses": doses,
        "nested_doses": True,
        "unique_sources_required": True,
        "training_authorized": False,
    }

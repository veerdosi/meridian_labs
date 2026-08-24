"""Deterministic selection of unique successful intervention trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _feature_value(record: Mapping[str, Any], feature: str) -> float:
    if "initial_physical_features" in record:
        return float(record["initial_physical_features"][feature])
    _, joint, axis = feature.split(":")
    return float(record["initial_free_joint_positions"][joint][("x", "y", "z").index(axis)])


def _on_failure_side(value: float, rule: Mapping[str, Any]) -> bool:
    return value <= float(rule["threshold"]) if rule["failure_side"] == "low" else value >= float(rule["threshold"])


def select_data_arms(
    selection: Mapping[str, Any],
    boundary_record: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict:
    selected = selection.get("selected")
    if not selected:
        raise ValueError("no selected boundary")
    screening = [
        record
        for record in candidate_records
        if record.get("parameters", {}).get("phase") == "screening"
    ]
    sources = [
        record
        for record in candidate_records
        if record.get("parameters", {}).get("phase") == "training_source_pool"
    ]
    expected_screen = {
        (task["suite"], int(task["task_id"]), int(init_state))
        for task in config["task_set"]
        for init_state in config["partitions"]["screening_init_states"]
    }
    observed_screen = {
        (record["task_suite"], int(record["task_id"]), int(record["init_state_index"]))
        for record in screening
    }
    if observed_screen != expected_screen or len(screening) != len(expected_screen):
        raise ValueError("random source universe does not contain the complete locked screening set")
    expected_sources = {
        (selected["task_suite"], int(selected["task_id"]), int(init_state))
        for init_state in config["partitions"]["training_init_states"]
    }
    observed_sources = {
        (record["task_suite"], int(record["task_id"]), int(record["init_state_index"]))
        for record in sources
    }
    if observed_sources != expected_sources or len(sources) != len(expected_sources):
        raise ValueError("target source universe does not contain the complete locked source pool")
    successful_records = [record for record in candidate_records if record["success"]]
    unique = {str(record["id"]): record for record in successful_records}
    if len(unique) != len(successful_records):
        raise ValueError("source universe contains duplicate rollout IDs")
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
    rule = selected["coverage_rule"]
    feature = str(rule["feature"])
    boundary = _feature_value(boundary_record, feature)
    targeted = sorted(
        task_records,
        key=lambda record: (
            abs(_feature_value(record, feature) - boundary),
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
    targeted_coverage = {
        str(dose): sum(
            _on_failure_side(_feature_value(record, feature), rule)
            for record in targeted[:dose]
        )
        / dose
        for dose in sorted(int(value) for value in config["training"]["doses"])
    }
    return {
        "schema": "meridian-physical-data-selection-v1",
        "doses": doses,
        "nested_doses": True,
        "unique_sources_required": True,
        "target_feature": feature,
        "target_rule": rule,
        "targeted_failure_side_coverage": targeted_coverage,
    }

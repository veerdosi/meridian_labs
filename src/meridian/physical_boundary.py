"""Locked natural-initial-state boundary screening and selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SUITE_ORDER = {name: index for index, name in enumerate(
    ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
)}


def validate_protocol_config(config: Mapping[str, Any]) -> None:
    tasks = [(task["suite"], int(task["task_id"])) for task in config["task_set"]]
    if len(tasks) != len(set(tasks)):
        raise ValueError("task_set contains duplicates")
    if Counter(suite for suite, _ in tasks) != Counter({suite: 2 for suite in SUITE_ORDER}):
        raise ValueError("task_set must contain exactly two tasks from each LIBERO suite")
    partition_names = (
        "screening_init_states",
        "confirmation_init_states",
        "training_init_states",
        "untouched_holdout_init_states",
    )
    seen: set[int] = set()
    for name in partition_names:
        values = [int(value) for value in config["partitions"][name]]
        if len(values) != len(set(values)) or any(value < 0 or value >= 50 for value in values):
            raise ValueError(f"partition {name} has duplicate or out-of-range indices")
        if overlap := seen & set(values):
            raise ValueError(f"initial-state partitions overlap at {sorted(overlap)}")
        seen.update(values)
    if not np.isclose(sum(float(value) for value in config["weights"].values()), 1.0):
        raise ValueError("selection weights must sum to one")
    evaluation = config["evaluation"]
    if int(evaluation["target_trials"]) != int(evaluation["target_holdout_states"]) * int(
        evaluation["target_repeats_per_holdout"]
    ):
        raise ValueError("target trial count does not match holdout count times repeats")
    if int(evaluation["regression_trials"]) != len(evaluation["regression_tasks"]) * len(
        evaluation["regression_states_per_task"]
    ):
        raise ValueError("regression trial count does not match its task/state product")
    doses = [int(value) for value in config["training"]["doses"]]
    if doses != sorted(set(doses)) or not doses or doses[0] <= 0:
        raise ValueError("training doses must be unique positive values in ascending order")
    if int(config["thresholds"]["minimum_unique_targeted_sources"]) < max(doses):
        raise ValueError("unique targeted-source threshold must cover the largest dose")


def make_screening_plans(config: Mapping[str, Any]) -> list[dict]:
    settings = config["screening"]
    plans = []
    for task_index, task in enumerate(config["task_set"]):
        for state_index, init_state in enumerate(config["partitions"]["screening_init_states"]):
            plans.append(
                {
                    "id": f"screen-{task['suite']}-t{task['task_id']}-i{init_state}",
                    "task_suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "seed": int(settings["seed"]) + task_index * 100 + state_index,
                    "init_state_index": int(init_state),
                    "replan_steps": int(settings["replan_steps"]),
                    "wait_steps": int(settings["wait_steps"]),
                    "phase": "screening",
                }
            )
    return plans


def _feature_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    names = {name for record in records for name in record["initial_free_joint_positions"]}
    if not names:
        raise ValueError("rollouts contain no initial free-joint positions")
    if any(set(record["initial_free_joint_positions"]) != names for record in records):
        raise ValueError("physical object schema differs within one task")
    return sorted(names)


def _features(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if all("initial_sim_qpos" in record for record in records):
        values = np.asarray([record["initial_sim_qpos"] for record in records], dtype=float)
        if len({len(value) for value in values}) != 1:
            raise ValueError("initial qpos dimensions differ within one task")
        scale = np.std(values, axis=0)
        scale[scale < 1e-9] = 1.0
        return (values - np.mean(values, axis=0)) / scale
    names = _feature_names(records)
    values = np.asarray(
        [
            [coordinate for name in names for coordinate in record["initial_free_joint_positions"][name]]
            for record in records
        ],
        dtype=float,
    )
    scale = np.std(values, axis=0)
    scale[scale < 1e-9] = 1.0
    return (values - np.mean(values, axis=0)) / scale


def _geometry_summary(records: Sequence[Mapping[str, Any]]) -> dict:
    features = _features(records)
    outcomes = np.asarray([bool(record["success"]) for record in records])
    if outcomes.all() or (~outcomes).all():
        return {"specificity": 0.0, "boundary_failure_index": None, "random_coverage": 1.0}
    nearest_correct = 0
    for index in range(len(records)):
        distances = np.linalg.norm(features - features[index], axis=1)
        distances[index] = np.inf
        nearest_correct += bool(outcomes[int(np.argmin(distances))] == outcomes[index])
    success_indices, failure_indices = np.flatnonzero(outcomes), np.flatnonzero(~outcomes)
    pairs = [
        (float(np.linalg.norm(features[failure] - features[success])), failure, success)
        for failure in failure_indices
        for success in success_indices
    ]
    radius, boundary_failure, _ = min(pairs, key=lambda item: (item[0], item[1], item[2]))
    distances = np.linalg.norm(features - features[boundary_failure], axis=1)
    random_coverage = float(np.mean(distances < max(radius, 1e-9)))
    return {
        "specificity": nearest_correct / len(records),
        "boundary_failure_index": int(boundary_failure),
        "random_coverage": random_coverage,
    }


def summarize_screening(
    records: Sequence[Mapping[str, Any]], diagnoses: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["task_suite"]), int(record["task_id"]))].append(record)
    expected = len(config["partitions"]["screening_init_states"])
    thresholds = config["thresholds"]
    weights = config["weights"]
    summaries = []
    for (suite, task_id), members in grouped.items():
        members = sorted(members, key=lambda item: int(item["init_state_index"]))
        if len(members) != expected:
            raise ValueError(f"incomplete screening for {suite} task {task_id}: {len(members)}/{expected}")
        successes = sum(bool(record["success"]) for record in members)
        failures = len(members) - successes
        geometry = _geometry_summary(members)
        failed_stages = [
            diagnoses[record["id"]]["diagnosis"]["stage"]
            for record in members
            if not record["success"]
        ]
        stage_consistency = (
            max(Counter(failed_stages).values()) / len(failed_stages) if failed_stages else 0.0
        )
        physical_specificity = (geometry["specificity"] + stage_consistency) / 2
        competence = successes / len(members)
        headroom = min(1.0, successes / 3) * min(1.0, failures / 2)
        rarity = 1 - geometry["random_coverage"]
        partial_weight = (
            weights["canonical_competence"]
            + weights["intervention_headroom"]
            + weights["physical_specificity"]
            + weights["random_rarity"]
        )
        partial_score = (
            weights["canonical_competence"] * competence
            + weights["intervention_headroom"] * headroom
            + weights["physical_specificity"] * physical_specificity
            + weights["random_rarity"] * rarity
        ) / partial_weight
        boundary_record = (
            members[geometry["boundary_failure_index"]]
            if geometry["boundary_failure_index"] is not None
            else None
        )
        eligible = (
            successes >= int(thresholds["min_screen_successes"])
            and failures >= int(thresholds["min_screen_failures"])
            and physical_specificity >= float(thresholds["min_geometry_specificity"])
            and geometry["random_coverage"]
            <= float(thresholds["max_expected_random_target_coverage"])
        )
        summaries.append(
            {
                "task_suite": suite,
                "task_id": task_id,
                "screen_successes": successes,
                "screen_trials": len(members),
                "canonical_competence": competence,
                "intervention_headroom": headroom,
                "geometry_specificity": geometry["specificity"],
                "stage_consistency": stage_consistency,
                "physical_specificity": physical_specificity,
                "expected_random_target_coverage": geometry["random_coverage"],
                "dominant_failure_stage": Counter(failed_stages).most_common(1)[0][0] if failed_stages else None,
                "boundary_init_state_index": int(boundary_record["init_state_index"]) if boundary_record else None,
                "boundary_rollout_id": boundary_record["id"] if boundary_record else None,
                "partial_score": partial_score,
                "eligible_for_confirmation": eligible,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            not item["eligible_for_confirmation"],
            -item["partial_score"],
            -item["physical_specificity"],
            SUITE_ORDER[item["task_suite"]],
            item["task_id"],
            item["boundary_init_state_index"] if item["boundary_init_state_index"] is not None else 999,
        ),
    )


def make_confirmation_plans(summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict]:
    count = int(config["screening"]["top_candidates_to_confirm"])
    candidates = [item for item in summaries if item["eligible_for_confirmation"]][:count]
    settings = config["confirmation"]
    plans = []
    for candidate_index, candidate in enumerate(candidates):
        common = {
            "task_suite": candidate["task_suite"],
            "task_id": int(candidate["task_id"]),
            "init_state_index": int(candidate["boundary_init_state_index"]),
            "wait_steps": int(config["screening"]["wait_steps"]),
        }
        for repeat in range(int(settings["canonical_repeats"])):
            plans.append(
                {
                    "id": f"confirm-{candidate_index}-repeat-{repeat}",
                    **common,
                    "seed": int(settings["seed"]) + candidate_index * 100 + repeat,
                    "replan_steps": int(config["screening"]["replan_steps"]),
                    "repeat": repeat,
                    "phase": "repeatability",
                }
            )
        for probe_index, replan_steps in enumerate(settings["control_replan_steps"]):
            plans.append(
                {
                    "id": f"confirm-{candidate_index}-control-r{replan_steps}",
                    **common,
                    "seed": int(settings["seed"]) + candidate_index * 100 + 50 + probe_index,
                    "replan_steps": int(replan_steps),
                    "phase": "control_probe",
                }
            )
    return plans


def make_source_pool_plans(selection: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict]:
    selected = selection.get("selected")
    if not selected:
        raise ValueError("cannot plan source collection without a selected boundary")
    plans = []
    for offset, init_state in enumerate(config["partitions"]["training_init_states"]):
        plans.append(
            {
                "id": f"source-{selected['task_suite']}-t{selected['task_id']}-i{init_state}",
                "task_suite": selected["task_suite"],
                "task_id": int(selected["task_id"]),
                "seed": int(config["training"]["source_seed"]) + offset,
                "init_state_index": int(init_state),
                "replan_steps": int(config["screening"]["replan_steps"]),
                "wait_steps": int(config["screening"]["wait_steps"]),
                "phase": "training_source_pool",
            }
        )
    return plans


def make_evaluation_plans(
    selection: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    screening_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict]:
    """Choose target holdouts by geometry only, without reading their outcomes."""
    selected = selection.get("selected")
    if not selected:
        raise ValueError("cannot plan evaluation without a selected boundary")
    task_key = (selected["task_suite"], int(selected["task_id"]))
    task_screen = [
        record for record in screening_records
        if (record["task_suite"], int(record["task_id"])) == task_key
    ]
    boundary = next(
        record for record in task_screen if record["id"] == selected["boundary_rollout_id"]
    )
    use_qpos = all("initial_sim_qpos" in record for record in task_screen)
    names = [] if use_qpos else _feature_names(task_screen)

    def vector(record: Mapping[str, Any]) -> np.ndarray:
        if use_qpos:
            return np.asarray(record["initial_sim_qpos"], dtype=float)
        return np.asarray(
            [coordinate for name in names for coordinate in record["initial_free_joint_positions"][name]],
            dtype=float,
        )

    screen_values = np.asarray([vector(record) for record in task_screen])
    scale = np.std(screen_values, axis=0)
    scale[scale < 1e-9] = 1.0
    boundary_vector = vector(boundary)
    holdout_set = set(config["partitions"]["untouched_holdout_init_states"])
    holdouts = [
        record for record in inventory
        if (record["task_suite"], int(record["task_id"])) == task_key
        and int(record["init_state_index"]) in holdout_set
    ]
    if len(holdouts) != len(holdout_set):
        raise ValueError("state inventory is missing selected-task untouched holdouts")
    holdouts.sort(
        key=lambda record: (
            float(np.linalg.norm((vector(record) - boundary_vector) / scale)),
            int(record["init_state_index"]),
        )
    )
    evaluation = config["evaluation"]
    selected_holdouts = holdouts[: int(evaluation["target_holdout_states"])]
    plans = []
    for state_offset, state in enumerate(selected_holdouts):
        for repeat in range(int(evaluation["target_repeats_per_holdout"])):
            plans.append(
                {
                    "id": f"target-i{state['init_state_index']}-r{repeat}",
                    "task_suite": selected["task_suite"],
                    "task_id": int(selected["task_id"]),
                    "seed": int(evaluation["seed"]) + state_offset * 100 + repeat,
                    "init_state_index": int(state["init_state_index"]),
                    "replan_steps": int(config["screening"]["replan_steps"]),
                    "wait_steps": int(config["screening"]["wait_steps"]),
                    "phase": "untouched_evaluation",
                    "evaluation_suite": "target",
                    "repeat": repeat,
                }
            )
    regression_states = evaluation["regression_states_per_task"]
    for task_offset, task in enumerate(evaluation["regression_tasks"]):
        for state_offset, init_state in enumerate(regression_states):
            plans.append(
                {
                    "id": f"regression-{task['suite']}-t{task['task_id']}-i{init_state}",
                    "task_suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "seed": int(evaluation["seed"]) + 10000 + task_offset * 100 + state_offset,
                    "init_state_index": int(init_state),
                    "replan_steps": int(config["screening"]["replan_steps"]),
                    "wait_steps": int(config["screening"]["wait_steps"]),
                    "phase": "untouched_evaluation",
                    "evaluation_suite": "regression",
                }
            )
    if len(plans) != int(evaluation["target_trials"]) + int(evaluation["regression_trials"]):
        raise ValueError("evaluation trial counts do not match the locked configuration")
    return plans


def finalize_selection(
    summaries: Sequence[Mapping[str, Any]], confirmation_records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict:
    by_task: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in confirmation_records:
        by_task[(str(record["task_suite"]), int(record["task_id"]))].append(record)
    weights = config["weights"]
    validated = []
    for summary in summaries:
        if not summary["eligible_for_confirmation"]:
            continue
        records = by_task.get((summary["task_suite"], int(summary["task_id"])), [])
        repeats = [record for record in records if record["parameters"].get("phase") == "repeatability"]
        controls = [record for record in records if record["parameters"].get("phase") == "control_probe"]
        expected_repeats = int(config["confirmation"]["canonical_repeats"])
        expected_controls = len(config["confirmation"]["control_replan_steps"])
        if len(repeats) != expected_repeats or len(controls) != expected_controls:
            continue
        repeatability = sum(not record["success"] for record in repeats) / len(repeats)
        persistence = sum(not record["success"] for record in controls) / len(controls)
        total = (
            weights["canonical_competence"] * summary["canonical_competence"]
            + weights["repeatability"] * repeatability
            + weights["control_persistence"] * persistence
            + weights["intervention_headroom"] * summary["intervention_headroom"]
            + weights["physical_specificity"] * summary["physical_specificity"]
            + weights["random_rarity"] * (1 - summary["expected_random_target_coverage"])
        )
        passed = repeatability == 1.0 and persistence == 1.0
        validated.append({**summary, "repeatability": repeatability, "control_persistence": persistence, "total_score": total, "passed": passed})
    validated.sort(key=lambda item: (-item["total_score"], -item["repeatability"], -item["physical_specificity"], SUITE_ORDER[item["task_suite"]], item["task_id"], item["boundary_init_state_index"]))
    selected = next((item for item in validated if item["passed"]), None)
    return {
        "schema": "meridian-physical-boundary-selection-v1",
        "selected": selected,
        "validated_candidates": validated,
        "decision": "selected" if selected else "no_valid_boundary",
        "training_authorized": False,
        "note": "Selection never authorizes training; source sufficiency and human scientific review remain required.",
    }

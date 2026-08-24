"""Locked natural-initial-state boundary screening and selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from meridian.search import _wilson

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


def _physical_feature_mapping(record: Mapping[str, Any]) -> dict[str, float]:
    if "initial_physical_features" in record:
        return {str(name): float(value) for name, value in record["initial_physical_features"].items()}
    mapping = {}
    for name, position in record["initial_free_joint_positions"].items():
        for axis, value in zip(("x", "y", "z"), position):
            mapping[f"object:{name}:{axis}"] = float(value)
    return mapping


def _feature_table(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], np.ndarray]:
    mappings = [_physical_feature_mapping(record) for record in records]
    names = set(mappings[0]) if mappings else set()
    if not names or any(set(mapping) != names for mapping in mappings):
        raise ValueError("interpretable physical feature schema differs within one task")
    ordered = sorted(names)
    return ordered, np.asarray([[mapping[name] for name in ordered] for mapping in mappings])


def _geometry_summary(records: Sequence[Mapping[str, Any]]) -> dict:
    names, features = _feature_table(records)
    outcomes = np.asarray([bool(record["success"]) for record in records])
    if outcomes.all() or (~outcomes).all():
        return {
            "specificity": 0.0,
            "boundary_failure_index": None,
            "random_coverage": 1.0,
            "coverage_rule": None,
        }
    actual_failure = ~outcomes
    candidates = []
    for feature_index, name in enumerate(names):
        values = features[:, feature_index]
        unique = np.unique(values)
        for threshold in (unique[:-1] + unique[1:]) / 2:
            for side_order, side in enumerate(("low", "high")):
                predicted_failure = values <= threshold if side == "low" else values >= threshold
                accuracy = float(np.mean(predicted_failure == actual_failure))
                candidates.append((-accuracy, name, float(threshold), side_order, predicted_failure))
    if not candidates:
        return {
            "specificity": 0.0,
            "boundary_failure_index": None,
            "random_coverage": 1.0,
            "coverage_rule": None,
        }
    negative_accuracy, feature, threshold, side_order, predicted_failure = min(
        candidates, key=lambda item: item[:4]
    )
    side = ("low", "high")[side_order]
    feature_index = names.index(feature)
    values = features[:, feature_index]
    failure_indices = np.flatnonzero(actual_failure)
    boundary_failure = min(
        failure_indices,
        key=lambda index: (abs(float(values[index]) - threshold), int(index)),
    )
    return {
        "specificity": -negative_accuracy,
        "boundary_failure_index": int(boundary_failure),
        "random_coverage": float(np.mean(predicted_failure)),
        "coverage_rule": {
            "feature": feature,
            "threshold": threshold,
            "failure_side": side,
            "boundary_failure_value": float(values[boundary_failure]),
        },
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
                "screen_wilson_95": _wilson(successes, len(members)),
                "canonical_competence": competence,
                "intervention_headroom": headroom,
                "geometry_specificity": geometry["specificity"],
                "stage_consistency": stage_consistency,
                "physical_specificity": physical_specificity,
                "expected_random_target_coverage": geometry["random_coverage"],
                "coverage_rule": geometry["coverage_rule"],
                "dominant_failure_stage": Counter(failed_stages).most_common(1)[0][0] if failed_stages else None,
                "boundary_init_state_index": int(boundary_record["init_state_index"]) if boundary_record else None,
                "boundary_rollout_id": boundary_record["id"] if boundary_record else None,
                "evidence_rollout_ids": [record["id"] for record in members],
                "failure_rollout_ids": [record["id"] for record in members if not record["success"]],
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


def build_physical_capability_map(
    summaries: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict:
    successes = sum(bool(record["success"]) for record in records)
    return {
        "schema": "meridian-physical-capability-map-v1",
        "rollout_count": len(records),
        "successes": successes,
        "global_success_rate": successes / len(records),
        "global_wilson_95": _wilson(successes, len(records)),
        "task_regions": list(summaries),
        "eligible_boundary_count": sum(
            bool(item["eligible_for_confirmation"]) for item in summaries
        ),
        "evidence_rollout_ids": [record["id"] for record in records],
    }


def _predicted_failure(record: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    value = _physical_feature_mapping(record)[str(rule["feature"])]
    return value <= float(rule["threshold"]) if rule["failure_side"] == "low" else value >= float(rule["threshold"])


def make_confirmation_plans(
    summaries: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict]:
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
        confirmation_states = set(config["partitions"]["confirmation_init_states"])
        inventory_records = sorted(
            (
                record
                for record in inventory
                if record["task_suite"] == candidate["task_suite"]
                and int(record["task_id"]) == int(candidate["task_id"])
                and int(record["init_state_index"]) in confirmation_states
            ),
            key=lambda record: int(record["init_state_index"]),
        )
        if len(inventory_records) != len(confirmation_states):
            raise ValueError("state inventory is missing candidate confirmation states")
        if any(record.get("goal_already_satisfied") for record in inventory_records):
            raise ValueError("a confirmation initial state already satisfies the task goal")
        for state_offset, state in enumerate(inventory_records):
            predicted_failure = _predicted_failure(state, candidate["coverage_rule"])
            plans.append(
                {
                    "id": f"confirm-{candidate_index}-generalize-i{state['init_state_index']}",
                    "task_suite": candidate["task_suite"],
                    "task_id": int(candidate["task_id"]),
                    "seed": int(settings["seed"]) + candidate_index * 100 + 70 + state_offset,
                    "init_state_index": int(state["init_state_index"]),
                    "wait_steps": int(config["screening"]["wait_steps"]),
                    "replan_steps": int(config["screening"]["replan_steps"]),
                    "phase": (
                        "boundary_predicted_failure"
                        if predicted_failure
                        else "boundary_predicted_success"
                    ),
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
    rule = selected["coverage_rule"]
    feature = str(rule["feature"])
    boundary_value = _physical_feature_mapping(boundary)[feature]
    holdout_set = set(config["partitions"]["untouched_holdout_init_states"])
    holdouts = [
        record for record in inventory
        if (record["task_suite"], int(record["task_id"])) == task_key
        and int(record["init_state_index"]) in holdout_set
    ]
    if len(holdouts) != len(holdout_set):
        raise ValueError("state inventory is missing selected-task untouched holdouts")
    target_holdouts = [record for record in holdouts if _predicted_failure(record, rule)]
    target_count = int(config["evaluation"]["target_holdout_states"])
    if len(target_holdouts) < target_count:
        raise ValueError(
            f"only {len(target_holdouts)} untouched states match the locked failure side; "
            f"{target_count} required"
        )
    target_holdouts.sort(
        key=lambda record: (
            abs(_physical_feature_mapping(record)[feature] - boundary_value),
            int(record["init_state_index"]),
        )
    )
    evaluation = config["evaluation"]
    selected_holdouts = target_holdouts[:target_count]
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
    summaries: Sequence[Mapping[str, Any]],
    confirmation_records: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
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
        predicted_failures = [
            record
            for record in records
            if record["parameters"].get("phase") == "boundary_predicted_failure"
        ]
        predicted_successes = [
            record
            for record in records
            if record["parameters"].get("phase") == "boundary_predicted_success"
        ]
        expected_repeats = int(config["confirmation"]["canonical_repeats"])
        expected_controls = len(config["confirmation"]["control_replan_steps"])
        expected_generalization = len(config["partitions"]["confirmation_init_states"])
        if (
            len(repeats) != expected_repeats
            or len(controls) != expected_controls
            or len(predicted_failures) + len(predicted_successes) != expected_generalization
        ):
            continue
        repeatability = sum(not record["success"] for record in repeats) / len(repeats)
        persistence = sum(not record["success"] for record in controls) / len(controls)
        if predicted_failures and predicted_successes:
            predicted_failure_success_rate = sum(
                bool(record["success"]) for record in predicted_failures
            ) / len(predicted_failures)
            predicted_success_success_rate = sum(
                bool(record["success"]) for record in predicted_successes
            ) / len(predicted_successes)
            generalization_contrast = (
                predicted_success_success_rate - predicted_failure_success_rate
            )
        else:
            predicted_failure_success_rate = None
            predicted_success_success_rate = None
            generalization_contrast = -1.0
        validated_specificity = max(
            0.0, (summary["physical_specificity"] + generalization_contrast) / 2
        )
        holdout_indices = set(config["partitions"]["untouched_holdout_init_states"])
        matching_holdouts = [
            record
            for record in inventory
            if record["task_suite"] == summary["task_suite"]
            and int(record["task_id"]) == int(summary["task_id"])
            and int(record["init_state_index"]) in holdout_indices
            and _predicted_failure(record, summary["coverage_rule"])
        ]
        required_holdouts = int(config["evaluation"]["target_holdout_states"])
        total = (
            weights["canonical_competence"] * summary["canonical_competence"]
            + weights["repeatability"] * repeatability
            + weights["control_persistence"] * persistence
            + weights["intervention_headroom"] * summary["intervention_headroom"]
            + weights["physical_specificity"] * validated_specificity
            + weights["random_rarity"] * (1 - summary["expected_random_target_coverage"])
        )
        passed = (
            repeatability == 1.0
            and persistence == 1.0
            and generalization_contrast
            >= float(config["thresholds"]["min_generalization_contrast"])
            and len(matching_holdouts) >= required_holdouts
        )
        validated.append(
            {
                **summary,
                "repeatability": repeatability,
                "control_persistence": persistence,
                "predicted_failure_success_rate": predicted_failure_success_rate,
                "predicted_success_success_rate": predicted_success_success_rate,
                "generalization_contrast": generalization_contrast,
                "validated_physical_specificity": validated_specificity,
                "matching_untouched_holdouts": len(matching_holdouts),
                "total_score": total,
                "passed": passed,
            }
        )
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

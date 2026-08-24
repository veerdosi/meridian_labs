from pathlib import Path

import pytest
import yaml

from meridian.physical_boundary import (
    build_physical_capability_map,
    finalize_selection,
    make_confirmation_plans,
    make_evaluation_plans,
    make_screening_plans,
    summarize_screening,
    validate_protocol_config,
)


def config() -> dict:
    return {
        "task_set": [{"suite": "libero_object", "task_id": 0}],
        "partitions": {
            "screening_init_states": [0, 1, 2, 3, 4, 5],
            "confirmation_init_states": [10, 11, 12, 13],
            "untouched_holdout_init_states": [40, 41, 42, 43],
        },
        "screening": {"seed": 10, "replan_steps": 5, "wait_steps": 10, "top_candidates_to_confirm": 1},
        "confirmation": {"canonical_repeats": 3, "control_replan_steps": [1, 10], "seed": 20},
        "thresholds": {"min_screen_successes": 3, "min_screen_failures": 1, "min_geometry_specificity": 0.5, "max_expected_random_target_coverage": 0.35, "min_generalization_contrast": 0.5},
        "weights": {"canonical_competence": 0.2, "repeatability": 0.2, "control_persistence": 0.2, "intervention_headroom": 0.15, "physical_specificity": 0.15, "random_rarity": 0.1},
        "evaluation": {"target_holdout_states": 4},
    }


def screening_records() -> tuple[list[dict], dict]:
    records, diagnoses = [], {}
    outcomes = [True, True, True, True, False, False]
    positions = [0.0, 0.1, 0.2, 0.3, 2.0, 2.1]
    for index, (success, position) in enumerate(zip(outcomes, positions)):
        identifier = f"r{index}"
        records.append({"id": identifier, "task_suite": "libero_object", "task_id": 0, "init_state_index": index, "success": success, "initial_free_joint_positions": {"object": [position, 0.0, 0.0]}})
        diagnoses[identifier] = {"diagnosis": {"stage": "complete" if success else "approach"}}
    return records, diagnoses


def confirmation_inventory() -> list[dict]:
    return [
        {
            "task_suite": "libero_object",
            "task_id": 0,
            "init_state_index": index,
            "initial_free_joint_positions": {"object": [position, 0.0, 0.0]},
            "goal_already_satisfied": False,
        }
        for index, position in (
            (10, 0.1),
            (11, 0.2),
            (12, 2.0),
            (13, 2.2),
            (40, 2.0),
            (41, 2.1),
            (42, 2.2),
            (43, 2.3),
        )
    ]


def test_plan_is_balanced_and_deterministic() -> None:
    plans = make_screening_plans(config())
    assert len(plans) == 6
    assert plans[0]["id"] == "screen-libero_object-t0-i0"
    assert [plan["init_state_index"] for plan in plans] == [0, 1, 2, 3, 4, 5]


def test_screening_requires_competence_and_physical_specificity() -> None:
    records, diagnoses = screening_records()
    summaries = summarize_screening(records, diagnoses, config())
    assert summaries[0]["eligible_for_confirmation"] is True
    assert summaries[0]["boundary_init_state_index"] in {4, 5}
    assert summaries[0]["dominant_failure_stage"] == "approach"
    assert summaries[0]["coverage_rule"]["feature"] == "object:object:x"
    capability_map = build_physical_capability_map(summaries, records)
    assert capability_map["rollout_count"] == 6
    assert len(capability_map["global_wilson_95"]) == 2
    assert capability_map["task_regions"][0]["failure_rollout_ids"] == ["r4", "r5"]


def test_confirmation_persistence_selects_but_never_authorizes_training() -> None:
    records, diagnoses = screening_records()
    summaries = summarize_screening(records, diagnoses, config())
    plans = make_confirmation_plans(summaries, confirmation_inventory(), config())
    confirmation = [
        {
            **plan,
            "success": plan["phase"] == "boundary_predicted_success",
            "parameters": {"phase": plan["phase"]},
        }
        for plan in plans
    ]
    result = finalize_selection(summaries, confirmation, confirmation_inventory(), config())
    assert result["decision"] == "selected"
    assert result["training_authorized"] is False


def test_control_fix_rejects_data_boundary() -> None:
    records, diagnoses = screening_records()
    summaries = summarize_screening(records, diagnoses, config())
    plans = make_confirmation_plans(summaries, confirmation_inventory(), config())
    confirmation = [
        {
            **plan,
            "success": plan["phase"] in {"control_probe", "boundary_predicted_success"},
            "parameters": {"phase": plan["phase"]},
        }
        for plan in plans
    ]
    assert finalize_selection(
        summaries, confirmation, confirmation_inventory(), config()
    )["decision"] == "no_valid_boundary"


def test_unreplicated_feature_boundary_is_rejected() -> None:
    records, diagnoses = screening_records()
    summaries = summarize_screening(records, diagnoses, config())
    plans = make_confirmation_plans(summaries, confirmation_inventory(), config())
    confirmation = [
        {
            **plan,
            "success": False,
            "parameters": {"phase": plan["phase"]},
        }
        for plan in plans
    ]
    assert finalize_selection(
        summaries, confirmation, confirmation_inventory(), config()
    )["decision"] == "no_valid_boundary"


def test_evaluation_selects_holdouts_from_geometry_without_outcomes() -> None:
    records, _ = screening_records()
    for record in records:
        record["initial_sim_qpos"] = [record["initial_free_joint_positions"]["object"][0]]
    selection = {
        "selected": {
            "task_suite": "libero_object",
            "task_id": 0,
            "boundary_rollout_id": "r4",
            "coverage_rule": {
                "feature": "object:object:x",
                "threshold": 1.15,
                "failure_side": "high",
                "boundary_failure_value": 2.0,
            },
        }
    }
    inventory = [
        {
            "task_suite": "libero_object",
            "task_id": 0,
            "init_state_index": index,
            "initial_sim_qpos": [position],
            "initial_free_joint_positions": {"object": [position, 0.0, 0.0]},
        }
        for index, position in ((40, 2.05), (41, 3.0))
    ]
    cfg = config()
    cfg["partitions"]["untouched_holdout_init_states"] = [40, 41]
    cfg["evaluation"] = {
        "target_holdout_states": 2,
        "target_repeats_per_holdout": 2,
        "target_trials": 4,
        "regression_trials": 1,
        "seed": 30,
        "regression_tasks": [{"suite": "libero_goal", "task_id": 1}],
        "regression_states_per_task": [40],
    }
    plans = make_evaluation_plans(selection, inventory, records, cfg)
    assert len([plan for plan in plans if plan["evaluation_suite"] == "target"]) == 4
    assert {plan["init_state_index"] for plan in plans[:4]} == {40, 41}


def test_versioned_protocol_is_internally_consistent() -> None:
    path = Path(__file__).parents[1] / "configs" / "physical_boundary_v1.yaml"
    validate_protocol_config(yaml.safe_load(path.read_text()))


def test_protocol_rejects_partition_leakage() -> None:
    path = Path(__file__).parents[1] / "configs" / "physical_boundary_v1.yaml"
    locked = yaml.safe_load(path.read_text())
    locked["partitions"]["training_init_states"][0] = locked["partitions"][
        "screening_init_states"
    ][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol_config(locked)

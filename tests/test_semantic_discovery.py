from pathlib import Path

from meridian.semantic_discovery import (
    build_discovery_plans,
    build_semantic_validation_plans,
    inventory_tasks,
    nominate_validation_candidates,
    rank_validated_boundaries,
    score_inventory,
    select_discovery_and_reserve,
)


def test_semantic_inventory_and_split_are_deterministic_and_disjoint(tmp_path: Path) -> None:
    task_map = tmp_path / "map.py"
    task_map.write_text(
        "libero_task_map = {'a': ['open_drawer', 'put_both_items_in_basket'], "
        "'b': ['stack_bowls', 'put_mug_left_of_plate']}\n"
    )
    for suite, names in {
        "a": ["open_drawer", "put_both_items_in_basket"],
        "b": ["stack_bowls", "put_mug_left_of_plate"],
    }.items():
        (tmp_path / suite).mkdir()
        for name in names:
            predicate = "(Open drawer_1)" if "open" in name else "(On item_1 target_1)"
            (tmp_path / suite / f"{name}.bddl").write_text(
                f"(:language {name.replace('_', ' ')})\n(:objects item_1 - item)\n"
                f"(:goal (And {predicate}))\n"
            )
    config = {
        "compatible_suites": ["a", "b"],
        "benchmark_discovery_priors": {
            "suites": {
                "a": {"failure_prior": 0.1, "reported": True},
                "b": {"failure_prior": 0.2, "reported": False},
            }
        },
        "selection": {
            "score_weights": {
                "interaction_complexity": 0.4, "semantic_rarity": 0.2,
                "benchmark_failure_prior": 0.2, "unreported_suite_uncertainty": 0.2,
            },
            "discovery_quota": {"a": 1, "b": 1},
            "confirmation_quota": {"a": 1, "b": 1},
            "discovery_count": 2,
            "confirmation_reserve_count": 2,
            "greedy_weights": {"prior_score": 0.65, "categorical_diversity": 0.35},
        },
        "discovery_screen": {
            "profiles": ["canonical", "compound_view_visual"],
            "canonical": {"replan_steps": 5.0, "action_noise": 0.0},
            "compound_view_visual": {"replan_steps": 5.0, "action_noise": 0.0},
            "seed_base": 100,
        },
    }
    inventory = inventory_tasks(task_map, tmp_path, config["compatible_suites"])
    scored = score_inventory(inventory, config)
    discovery, reserve = select_discovery_and_reserve(scored, config)
    assert len(inventory) == 4
    assert {(row["task_suite"], row["task_id"]) for row in discovery}.isdisjoint(
        {(row["task_suite"], row["task_id"]) for row in reserve}
    )
    plans = build_discovery_plans(discovery, config)
    assert len(plans) == 4
    for index in range(0, len(plans), 2):
        assert plans[index]["seed"] == plans[index + 1]["seed"]
        assert plans[index]["init_state_index"] == plans[index + 1]["init_state_index"]


def test_semantic_boundary_requires_repeated_canonical_control_failure() -> None:
    candidate = {
        "task_suite": "libero_90", "task_id": 3,
        "semantic_capability": "articulated_sequence", "discovery_prior_score": 0.8,
    }
    config = {
        "discovery_screen": {
            "canonical": {"camera_x": 0.0, "camera_yaw_deg": 0.0, "brightness": 1.0,
                          "occlusion": 0.0, "visual_distractors": 0.0,
                          "replan_steps": 5.0, "action_noise": 0.0},
            "compound_view_visual": {"camera_x": 0.1, "camera_yaw_deg": 35.0,
                                     "brightness": 1.0, "occlusion": 0.22,
                                     "visual_distractors": 4.0,
                                     "replan_steps": 5.0, "action_noise": 0.0},
        },
        "validation_gate": {
            "candidates": 6, "repeats": 5, "initial_state_offsets": [0, 11, 23, 37, 49],
            "seed_base": 61000,
            "required_conditions": ["canonical", "compound_view_visual_canonical_control",
                                    "viewpoint_only", "visual_only"],
            "minimum_canonical_success_rate": 0.6, "minimum_stress_failure_rate": 0.4,
            "minimum_intervention_headroom": 0.2,
            "canonical_control": {"replan_steps": 5.0, "action_noise": 0.0},
            "low_random_coverage_prior": 0.95,
            "score_weights": {"repeatability": 0.3, "intervention_headroom": 0.25,
                              "canonical_control_persistence": 0.2,
                              "coverage_specificity": 0.15, "low_random_coverage": 0.1},
        },
    }
    screen = [
        {"task_suite": "libero_90", "task_id": 3, "success": True,
         "parameters": {"stress_profile": "canonical"}},
        {"task_suite": "libero_90", "task_id": 3, "success": False,
         "parameters": {"stress_profile": "compound_view_visual"}},
    ]
    nominees = nominate_validation_candidates(screen, [candidate], config)
    assert len(nominees) == 1
    plans = build_semantic_validation_plans(nominees, config)
    assert len(plans) == 20
    results = []
    for plan in plans:
        condition = plan["validation_condition"]
        repeat = plan["seed"] % 100
        success = condition != "compound_view_visual_canonical_control" or repeat >= 3
        if condition in {"viewpoint_only", "visual_only"}:
            success = True
        results.append({
            "task_suite": plan["task_suite"], "task_id": plan["task_id"],
            "success": success,
            "parameters": {"validation_condition": condition,
                           "replan_steps": plan["replan_steps"],
                           "action_noise": plan["action_noise"]},
        })
    ranked = rank_validated_boundaries(results, nominees, config)
    assert ranked[0]["eligible"]
    assert ranked[0]["stress_failure_rate"] == 0.6
    assert ranked[0]["canonical_control_failure_rate"] == 0.6
    assert ranked[0]["coverage_specificity"] == 0.6

from meridian.multitask import (
    build_multitask_screen_plans,
    build_multitask_validation_plans,
    nominate_multitask_candidates,
    score_validated_candidates,
)


def test_multitask_screen_is_paired_and_keeps_control_canonical() -> None:
    config = {
        "canonical": {"camera_x": 0.0, "replan_steps": 5.0, "action_noise": 0.0},
        "profiles": {"canonical": {}, "viewpoint": {"camera_x": 0.1}},
        "screen": {"repeats": 1},
        "suites": [
            {"name": "libero_spatial", "task_ids": [0, 1]},
            {"name": "libero_object", "task_ids": [0, 1]},
        ],
    }
    plans = build_multitask_screen_plans(config)
    assert set(plans) == {"libero_spatial", "libero_object"}
    assert all(len(suite_plans) == 4 for suite_plans in plans.values())
    for suite_plans in plans.values():
        assert all(plan["replan_steps"] == 5.0 for plan in suite_plans)
        assert all(plan["action_noise"] == 0.0 for plan in suite_plans)
        paired = {(plan["task_id"], plan["stress_profile"]) for plan in suite_plans}
        assert paired == {(0, "canonical"), (0, "viewpoint"), (1, "canonical"), (1, "viewpoint")}


def test_nomination_requires_canonical_success_and_is_not_selection() -> None:
    config = {
        "canonical": {"replan_steps": 5.0, "action_noise": 0.0},
        "validation": {"candidates": 8},
        "nomination": {
            "maximum_per_suite_before_global_fill": 2,
            "profile_priors": {
                "viewpoint": {"coverage_specificity": 0.8, "low_random_coverage": 0.7}
            },
        },
        "selection_criteria": {
            "intervention_headroom_weight": 0.25,
            "canonical_control_persistence_weight": 0.2,
            "coverage_specificity_weight": 0.15,
            "low_random_coverage_weight": 0.1,
        },
    }
    results = [
        {"task_suite": "libero_goal", "task_id": 0, "task": "a", "success": True,
         "parameters": {"stress_profile": "canonical"}},
        {"task_suite": "libero_goal", "task_id": 0, "task": "a", "success": False,
         "parameters": {"stress_profile": "viewpoint", "replan_steps": 5, "action_noise": 0}},
        {"task_suite": "libero_goal", "task_id": 1, "task": "b", "success": False,
         "parameters": {"stress_profile": "canonical"}},
        {"task_suite": "libero_goal", "task_id": 1, "task": "b", "success": False,
         "parameters": {"stress_profile": "viewpoint", "replan_steps": 5, "action_noise": 0}},
    ]
    nominees = nominate_multitask_candidates(results, config)
    assert [(row["task_id"], row["selection_status"]) for row in nominees] == [
        (0, "validation_nominee_only")
    ]


def test_validation_plans_pair_seed_and_init_across_conditions() -> None:
    config = {
        "canonical": {"camera_x": 0.0, "replan_steps": 5.0, "action_noise": 0.0},
        "profiles": {
            "canonical": {},
            "viewpoint": {"camera_x": 0.1},
            "visual": {"occlusion": 0.2},
            "compound_view_visual": {"camera_x": 0.1, "occlusion": 0.2},
        },
        "validation": {"initial_state_offsets": [0, 11, 23, 37, 49]},
    }
    plans = build_multitask_validation_plans(
        [{"task_suite": "libero_10", "task_id": 2, "candidate_profile": "compound_view_visual"}],
        config,
    )
    assert len(plans) == 20
    for repeat in range(5):
        paired = [plan for plan in plans if plan["seed"] == 20000 + repeat]
        assert len({plan["seed"] for plan in paired}) == 1
        assert len({plan["init_state_index"] for plan in paired}) == 1
        assert {plan["validation_condition"] for plan in paired} == {
            "canonical", "compound_view_visual", "viewpoint", "visual"
        }
        assert all(plan["replan_steps"] == 5.0 and plan["action_noise"] == 0.0 for plan in paired)


def test_locked_validation_score_accepts_repeatable_specific_candidate() -> None:
    config = {
        "nomination": {"profile_priors": {"viewpoint": {"low_random_coverage": 0.7}}},
        "selection_criteria": {
            "repeatability_weight": 0.3, "intervention_headroom_weight": 0.25,
            "canonical_control_persistence_weight": 0.2, "coverage_specificity_weight": 0.15,
            "low_random_coverage_weight": 0.1, "minimum_validation_repeats": 5,
            "minimum_canonical_success_rate": 0.6, "minimum_stress_failure_rate": 0.4,
            "minimum_intervention_headroom": 0.2, "minimum_canonical_control_failure_rate": 0.4,
        },
    }
    candidate = {"task_suite": "libero_goal", "task_id": 3, "candidate_profile": "viewpoint"}
    results = []
    for repeat in range(5):
        for condition, success in (("canonical", True), ("viewpoint", repeat >= 3), ("visual", True)):
            results.append({
                "task_suite": "libero_goal", "task_id": 3, "success": success,
                "parameters": {"candidate_profile": "viewpoint", "validation_condition": condition,
                               "replan_steps": 5.0, "action_noise": 0.0},
            })
    scored = score_validated_candidates(results, [candidate], config)[0]
    assert scored["eligible"]
    assert scored["stress_failure_rate"] == 0.6
    assert scored["intervention_headroom"] == 0.6
    assert scored["coverage_specificity"] == 0.6

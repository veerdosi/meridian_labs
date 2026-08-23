from meridian.multitask import build_multitask_screen_plans


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

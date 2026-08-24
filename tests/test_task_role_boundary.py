from meridian.task_role_boundary import assess_role_confirmation


def test_role_confirmation_requires_both_tasks_and_pooled_mechanism() -> None:
    config = {
        "tasks": [
            {"suite": "libero_90", "task_id": 18},
            {"suite": "libero_90", "task_id": 37},
        ],
        "confirmation": {"trials": 16},
        "confirmation_gate": {
            "object_motion_threshold_m": 0.025,
            "require_each_task_overall_success_at_most": 0.125,
            "require_each_task_wrong_object_fraction_at_least": 0.75,
            "require_pooled_wrong_object_trials_at_least": 12,
            "require_each_task_extended_horizon_successes": 0,
            "require_each_task_replanning_control_successes_at_most": 1,
            "human_review_required": True,
        },
    }
    phases = [
        "canonical",
        "canonical",
        "rapid_replan",
        "rapid_replan",
        "long_chunk",
        "long_chunk",
        "extended_horizon",
        "extended_horizon",
    ]
    rollouts = []
    diagnoses = []
    for task_id in (18, 37):
        for index, phase in enumerate(phases):
            identifier = f"t{task_id}-{index}"
            rollouts.append(
                {
                    "id": identifier,
                    "task_suite": "libero_90",
                    "task_id": task_id,
                    "success": False,
                    "parameters": {"phase": f"role_confirmation_{phase}"},
                }
            )
            diagnoses.append(
                {
                    "id": identifier,
                    "metrics": {
                        "max_target_object_translation_m": 0.0,
                        "max_distractor_translation_m": 0.1 if index < 6 else 0.0,
                    },
                }
            )
    result = assess_role_confirmation(rollouts, diagnoses, config)
    assert result["automated_pass"] is True
    assert result["pooled_wrong_object_trials"] == 12
    assert result["decision"] == "requires_human_review"


def test_role_confirmation_rejects_if_longer_horizon_solves_a_task() -> None:
    config = {
        "tasks": [{"suite": "libero_90", "task_id": 18}],
        "confirmation": {"trials": 2},
        "confirmation_gate": {
            "object_motion_threshold_m": 0.025,
            "require_each_task_overall_success_at_most": 0.5,
            "require_each_task_wrong_object_fraction_at_least": 0.0,
            "require_pooled_wrong_object_trials_at_least": 0,
            "require_each_task_extended_horizon_successes": 0,
            "require_each_task_replanning_control_successes_at_most": 1,
            "human_review_required": True,
        },
    }
    rollouts = [
        {
            "id": f"r{index}",
            "task_suite": "libero_90",
            "task_id": 18,
            "success": success,
            "parameters": {"phase": phase},
        }
        for index, (success, phase) in enumerate(
            [(False, "role_confirmation_canonical"), (True, "role_confirmation_extended_horizon")]
        )
    ]
    diagnoses = [
        {
            "id": f"r{index}",
            "metrics": {
                "max_target_object_translation_m": 0.0,
                "max_distractor_translation_m": 0.1,
            },
        }
        for index in range(2)
    ]
    assert assess_role_confirmation(rollouts, diagnoses, config)["automated_pass"] is False

from meridian.gating import evaluate_dose_gate, evaluate_sequential_gate


def records(target: list[bool], regression: list[bool]) -> list[dict]:
    return [
        {
            "id": f"target-{index}",
            "success": success,
            "task_suite": "target_suite",
            "task_id": 0,
            "parameters": {"evaluation_suite": "target"},
        }
        for index, success in enumerate(target)
    ] + [
        {
            "id": f"regression-{index}",
            "success": success,
            "task_suite": "regression_suite",
            "task_id": 1,
            "parameters": {"evaluation_suite": "regression:test"},
        }
        for index, success in enumerate(regression)
    ]


def test_gate_is_negative_on_targeted_tie() -> None:
    decision = evaluate_sequential_gate(
        baseline=records([False, False, True, True], [True, True]),
        targeted=records([True, False, True, True], [True, True]),
        random=records([True, True, False, True], [True, True]),
        regression_limit=0.02,
        expected_target=4,
        expected_regression=2,
    )
    assert decision["gate"] == "negative"
    assert decision["paired_targeted_vs_random"] == {
        "wins": 1,
        "losses": 1,
        "ties": 2,
        "discordant": 2,
        "two_sided_sign_p": 1.0,
    }


def test_gate_requires_regression_safety() -> None:
    decision = evaluate_sequential_gate(
        baseline=records([False, False, True, True], [True, True]),
        targeted=records([True, True, True, True], [True, False]),
        random=records([False, True, True, True], [True, True]),
        regression_limit=0.02,
        expected_target=4,
        expected_regression=2,
    )
    assert decision["gate"] == "negative"
    assert decision["targeted_regression_loss"] == 0.5


def test_physical_dose_releases_original_only_after_beating_random() -> None:
    decision = evaluate_dose_gate(
        baseline=records([False, False, True, True], [True, True]),
        targeted=records([True, True, True, True], [True, True]),
        random=records([False, True, True, True], [True, True]),
        original=None,
        regression_limit=0.02,
        dose=4,
        expected_target=4,
        expected_regression=2,
    )
    assert decision["decision"] == "release_original"
    task_summary = decision["arms"]["targeted"]["target_by_task"]["target_suite:0"]
    assert task_summary["successes"] == 4
    assert task_summary["trials"] == 4
    assert task_summary["success_rate"] == 1.0
    assert decision["paired"]["released_checkpoint"]["wins"] == 2


def test_physical_medium_dose_stops_when_marginal_gain_saturates() -> None:
    decision = evaluate_dose_gate(
        baseline=records([False, False, False, True], [True, True]),
        targeted=records([True, True, True, True], [True, True]),
        random=records([False, True, True, True], [True, True]),
        original=records([False, False, True, True], [True, True]),
        regression_limit=0.02,
        dose=8,
        prior_targeted={"target_successes": 4},
        expected_target=4,
        expected_regression=2,
    )
    assert decision["decision"] == "select_small_dose"

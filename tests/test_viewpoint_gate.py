from meridian.viewpoint_gate import evaluate_confirmation_gate, evaluate_primary_gate


def primary(target_successes: int, regression_successes: int) -> list[dict]:
    return [
        {
            "id": f"target-{index}",
            "success": index < target_successes,
            "parameters": {"evaluation_suite": "target"},
        }
        for index in range(40)
    ] + [
        {
            "id": f"regression-{index}",
            "success": index < regression_successes,
            "parameters": {"evaluation_suite": "regression:test"},
        }
        for index in range(40)
    ]


def confirmation(target_successes: int) -> list[dict]:
    return [
        {
            "id": f"confirmation-{index}",
            "success": index < target_successes,
            "parameters": {"evaluation_suite": "confirmation_target"},
        }
        for index in range(40)
    ]


def test_primary_pre_gate_releases_original_only_after_directional_win() -> None:
    decision = evaluate_primary_gate(
        baseline=primary(0, 40),
        targeted=primary(20, 39),
        random=primary(4, 40),
        original=None,
        stage="dose8",
    )
    assert decision["decision"] == "release_original"
    negative = evaluate_primary_gate(
        baseline=primary(0, 40),
        targeted=primary(4, 37),
        random=primary(4, 40),
        original=None,
        stage="dose8",
    )
    assert negative["decision"] == "negative_stop"


def test_complete_small_gate_is_decisive_or_escalates_without_posthoc_thresholds() -> None:
    decisive = evaluate_primary_gate(
        baseline=primary(0, 40),
        targeted=primary(20, 39),
        random=primary(4, 40),
        original=primary(2, 40),
        stage="dose8",
    )
    assert decisive["decision"] == "unlock_confirmation"
    assert decisive["comparisons"]["targeted_vs_random"]["lift"] == 0.4
    promising = evaluate_primary_gate(
        baseline=primary(0, 40),
        targeted=primary(5, 40),
        random=primary(1, 40),
        original=primary(0, 40),
        stage="dose8",
    )
    assert promising["decision"] == "escalate_medium_dose"
    medium = evaluate_primary_gate(
        baseline=primary(0, 40),
        targeted=primary(5, 40),
        random=primary(1, 40),
        original=primary(0, 40),
        stage="dose24",
    )
    assert medium["decision"] == "negative_stop"


def test_untouched_confirmation_requires_effect_and_paired_replication() -> None:
    decision = evaluate_confirmation_gate(
        primary_targeted=primary(20, 39),
        primary_random=primary(4, 40),
        confirmation_targeted=confirmation(18),
        confirmation_random=confirmation(2),
    )
    assert decision["decision"] == "supported"
    assert decision["confirmation_lift"] == 0.4

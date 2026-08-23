from meridian.confirmation import evaluate_harder_boundary_confirmation


def records(target_successes: set[int], regression_successes: int) -> list[dict]:
    rows = [
        {"id": f"target-{index}", "success": index in target_successes,
         "parameters": {"evaluation_suite": "target"}}
        for index in range(60)
    ]
    rows.extend(
        {"id": f"regression-{index}", "success": index < regression_successes,
         "parameters": {"evaluation_suite": "regression:libero_goal"}}
        for index in range(40)
    )
    return rows


def test_confirmation_requires_effect_significance_original_and_regression() -> None:
    thresholds = {
        "minimum_targeted_gain_over_random": 0.1,
        "targeted_vs_random_exact_p_max": 0.05,
        "minimum_targeted_gain_over_original_distribution": 0.1,
        "regression_noninferiority_margin_vs_released": 0.05,
    }
    result = evaluate_harder_boundary_confirmation(
        released=records(set(range(30)), 38),
        targeted=records(set(range(45)), 37),
        random=records(set(range(30)), 38),
        original=records(set(range(20)), 38),
        thresholds=thresholds,
    )
    assert result["confirmed"]
    assert result["paired_targeted_vs_random"]["wins"] == 15
    assert result["passes"] == {
        "minimum_targeted_gain_over_random": True,
        "paired_exact_p": True,
        "paired_direction": True,
        "minimum_targeted_gain_over_original_distribution": True,
        "regression_noninferiority": True,
    }

from __future__ import annotations

from meridian.gating import _arm_summary, _paired_sign_test


def evaluate_harder_boundary_confirmation(
    *,
    released: list[dict],
    targeted: list[dict],
    random: list[dict],
    original: list[dict],
    thresholds: dict,
    expected_target: int = 60,
    expected_regression: int = 40,
) -> dict:
    summaries = {
        "released_checkpoint": _arm_summary(
            released, expected_target=expected_target, expected_regression=expected_regression
        ),
        "targeted": _arm_summary(
            targeted, expected_target=expected_target, expected_regression=expected_regression
        ),
        "random": _arm_summary(
            random, expected_target=expected_target, expected_regression=expected_regression
        ),
        "original_distribution": _arm_summary(
            original, expected_target=expected_target, expected_regression=expected_regression
        ),
    }
    paired = _paired_sign_test(
        summaries["targeted"]["target_outcomes"], summaries["random"]["target_outcomes"]
    )
    targeted_gain_random = (
        summaries["targeted"]["target_success_rate"]
        - summaries["random"]["target_success_rate"]
    )
    targeted_gain_original = (
        summaries["targeted"]["target_success_rate"]
        - summaries["original_distribution"]["target_success_rate"]
    )
    regression_loss = (
        summaries["released_checkpoint"]["regression_success_rate"]
        - summaries["targeted"]["regression_success_rate"]
    )
    passes = {
        "minimum_targeted_gain_over_random": targeted_gain_random
        >= float(thresholds["minimum_targeted_gain_over_random"]),
        "paired_exact_p": paired["two_sided_sign_p"]
        <= float(thresholds["targeted_vs_random_exact_p_max"]),
        "paired_direction": paired["wins"] > paired["losses"],
        "minimum_targeted_gain_over_original_distribution": targeted_gain_original
        >= float(thresholds["minimum_targeted_gain_over_original_distribution"]),
        "regression_noninferiority": regression_loss
        <= float(thresholds["regression_noninferiority_margin_vs_released"]),
    }
    for summary in summaries.values():
        summary.pop("target_outcomes")
    return {
        "confirmed": all(passes.values()),
        "passes": passes,
        "arms": summaries,
        "paired_targeted_vs_random": paired,
        "targeted_gain_over_random": targeted_gain_random,
        "targeted_gain_over_original_distribution": targeted_gain_original,
        "targeted_regression_loss_vs_released": regression_loss,
        "thresholds": thresholds,
    }

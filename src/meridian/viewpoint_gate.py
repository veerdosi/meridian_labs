from __future__ import annotations

from meridian.gating import _paired_sign_test
from meridian.search import _wilson


def _suite(record: dict) -> str:
    return str(record.get("parameters", {}).get("evaluation_suite", ""))


def _summary(records: list[dict], *, target_suite: str, target_trials: int, regression_trials: int) -> dict:
    target = [record for record in records if _suite(record) == target_suite]
    regression = [record for record in records if _suite(record).startswith("regression:")]
    if len(target) != target_trials or len(regression) != regression_trials:
        raise ValueError(
            f"incomplete evaluation: target={len(target)}/{target_trials}, "
            f"regression={len(regression)}/{regression_trials}"
        )
    target_successes = sum(bool(record["success"]) for record in target)
    regression_successes = sum(bool(record["success"]) for record in regression)
    return {
        "target_successes": target_successes,
        "target_trials": len(target),
        "target_success_rate": target_successes / len(target),
        "target_wilson_95": _wilson(target_successes, len(target)),
        "regression_successes": regression_successes,
        "regression_trials": len(regression),
        "regression_success_rate": (
            regression_successes / len(regression) if regression else None
        ),
        "regression_wilson_95": (
            _wilson(regression_successes, len(regression)) if regression else None
        ),
        "target_outcomes": {record["id"]: bool(record["success"]) for record in target},
    }


def evaluate_primary_gate(
    *,
    baseline: list[dict],
    targeted: list[dict],
    random: list[dict],
    original: list[dict] | None,
    stage: str,
    regression_limit: float = 0.05,
    minimum_lift: float = 0.15,
    maximum_p: float = 0.05,
) -> dict:
    summaries = {
        "released_checkpoint": _summary(
            baseline, target_suite="target", target_trials=40, regression_trials=40
        ),
        "targeted": _summary(
            targeted, target_suite="target", target_trials=40, regression_trials=40
        ),
        "random": _summary(
            random, target_suite="target", target_trials=40, regression_trials=40
        ),
    }
    if original is not None:
        summaries["original_distribution"] = _summary(
            original, target_suite="target", target_trials=40, regression_trials=40
        )
    baseline_rate = summaries["released_checkpoint"]["target_success_rate"]
    targeted_rate = summaries["targeted"]["target_success_rate"]
    random_rate = summaries["random"]["target_success_rate"]
    regression_loss = (
        summaries["released_checkpoint"]["regression_success_rate"]
        - summaries["targeted"]["regression_success_rate"]
    )
    paired_random = _paired_sign_test(
        summaries["targeted"]["target_outcomes"], summaries["random"]["target_outcomes"]
    )
    pre_gate_positive = (
        targeted_rate > baseline_rate
        and targeted_rate > random_rate
        and regression_loss <= regression_limit
    )
    comparisons = {
        "targeted_vs_random": {
            "lift": targeted_rate - random_rate,
            **paired_random,
        }
    }
    if original is None:
        decision = "release_original" if pre_gate_positive else "negative_stop"
        rationale = (
            "Targeted strictly beat released and random within the regression limit."
            if pre_gate_positive
            else "Targeted failed the locked released/random directional or regression pre-gate."
        )
    else:
        original_rate = summaries["original_distribution"]["target_success_rate"]
        paired_original = _paired_sign_test(
            summaries["targeted"]["target_outcomes"],
            summaries["original_distribution"]["target_outcomes"],
        )
        comparisons["targeted_vs_original_distribution"] = {
            "lift": targeted_rate - original_rate,
            **paired_original,
        }
        directional = pre_gate_positive and targeted_rate > original_rate
        decisive = (
            directional
            and comparisons["targeted_vs_random"]["lift"] >= minimum_lift
            and comparisons["targeted_vs_original_distribution"]["lift"] >= minimum_lift
            and comparisons["targeted_vs_random"]["two_sided_sign_p"] <= maximum_p
            and comparisons["targeted_vs_original_distribution"]["two_sided_sign_p"]
            <= maximum_p
        )
        if decisive:
            decision = "unlock_confirmation"
            rationale = "Targeted passed both locked effect, paired-significance, and regression tests."
        elif directional and stage == "dose8":
            decision = "escalate_medium_dose"
            rationale = "Dose 8 was directionally positive but not decisive under the locked thresholds."
        else:
            decision = "negative_stop"
            rationale = "Targeted failed the complete locked gate; no post-hoc substitution is allowed."
    for summary in summaries.values():
        summary.pop("target_outcomes")
    return {
        "stage": stage,
        "decision": decision,
        "rationale": rationale,
        "thresholds": {
            "regression_limit": regression_limit,
            "minimum_lift": minimum_lift,
            "maximum_two_sided_sign_p": maximum_p,
        },
        "targeted_regression_loss": regression_loss,
        "arms": summaries,
        "comparisons": comparisons,
    }


def evaluate_confirmation_gate(
    *,
    primary_targeted: list[dict],
    primary_random: list[dict],
    confirmation_targeted: list[dict],
    confirmation_random: list[dict],
    minimum_lift: float = 0.15,
    maximum_confirmation_p: float = 0.05,
    maximum_combined_p: float = 0.01,
) -> dict:
    primary_t = _summary(
        primary_targeted, target_suite="target", target_trials=40, regression_trials=40
    )
    primary_r = _summary(
        primary_random, target_suite="target", target_trials=40, regression_trials=40
    )
    confirm_t = _summary(
        confirmation_targeted,
        target_suite="confirmation_target",
        target_trials=40,
        regression_trials=0,
    )
    confirm_r = _summary(
        confirmation_random,
        target_suite="confirmation_target",
        target_trials=40,
        regression_trials=0,
    )
    confirmation_pair = _paired_sign_test(
        confirm_t["target_outcomes"], confirm_r["target_outcomes"]
    )
    combined_targeted = {**primary_t["target_outcomes"], **confirm_t["target_outcomes"]}
    combined_random = {**primary_r["target_outcomes"], **confirm_r["target_outcomes"]}
    combined_pair = _paired_sign_test(combined_targeted, combined_random)
    lift = confirm_t["target_success_rate"] - confirm_r["target_success_rate"]
    passed = (
        lift >= minimum_lift
        and confirmation_pair["two_sided_sign_p"] <= maximum_confirmation_p
        and combined_pair["two_sided_sign_p"] <= maximum_combined_p
    )
    for summary in (primary_t, primary_r, confirm_t, confirm_r):
        summary.pop("target_outcomes")
    return {
        "decision": "supported" if passed else "not_confirmed",
        "confirmation_lift": lift,
        "confirmation_pair": confirmation_pair,
        "combined_primary_confirmation_pair": combined_pair,
        "thresholds": {
            "minimum_lift": minimum_lift,
            "maximum_confirmation_two_sided_sign_p": maximum_confirmation_p,
            "maximum_combined_two_sided_sign_p": maximum_combined_p,
        },
        "arms": {
            "primary_targeted": primary_t,
            "primary_random": primary_r,
            "confirmation_targeted": confirm_t,
            "confirmation_random": confirm_r,
        },
    }

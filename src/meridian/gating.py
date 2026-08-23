from __future__ import annotations

import math

from meridian.search import _wilson


def merge_rollouts(*groups: list[dict]) -> list[dict]:
    by_id = {}
    for record in (record for group in groups for record in group):
        previous = by_id.get(record["id"])
        if previous is not None and previous != record:
            raise ValueError(f"conflicting duplicate rollout: {record['id']}")
        by_id[record["id"]] = record
    return list(by_id.values())


def _paired_sign_test(candidate: dict[str, bool], reference: dict[str, bool]) -> dict:
    if candidate.keys() != reference.keys():
        raise ValueError("paired target rollout IDs differ")
    wins = sum(candidate[key] and not reference[key] for key in reference)
    losses = sum(reference[key] and not candidate[key] for key in reference)
    discordant = wins + losses
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    p_value = min(1.0, 2 * tail / (2**discordant)) if discordant else 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(reference) - discordant,
        "discordant": discordant,
        "two_sided_sign_p": p_value,
    }


def _arm_summary(records: list[dict], *, expected_target: int, expected_regression: int) -> dict:
    target = [
        record
        for record in records
        if record["parameters"].get("evaluation_suite") == "target"
    ]
    regression = [record for record in records if record not in target]
    if len(target) != expected_target or len(regression) != expected_regression:
        raise ValueError(
            f"incomplete evaluation: target={len(target)}/{expected_target}, "
            f"regression={len(regression)}/{expected_regression}"
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
        "regression_success_rate": regression_successes / len(regression),
        "target_outcomes": {record["id"]: bool(record["success"]) for record in target},
    }


def evaluate_sequential_gate(
    *,
    baseline: list[dict],
    targeted: list[dict],
    random: list[dict],
    regression_limit: float,
    expected_target: int = 40,
    expected_regression: int = 20,
) -> dict:
    summaries = {
        "released_checkpoint": _arm_summary(
            baseline, expected_target=expected_target, expected_regression=expected_regression
        ),
        "targeted_dose_8": _arm_summary(
            targeted, expected_target=expected_target, expected_regression=expected_regression
        ),
        "random_dose_8": _arm_summary(
            random, expected_target=expected_target, expected_regression=expected_regression
        ),
    }
    baseline_summary = summaries["released_checkpoint"]
    targeted_summary = summaries["targeted_dose_8"]
    random_summary = summaries["random_dose_8"]
    regression_loss = (
        baseline_summary["regression_success_rate"]
        - targeted_summary["regression_success_rate"]
    )
    paired = _paired_sign_test(
        targeted_summary.pop("target_outcomes"), random_summary.pop("target_outcomes")
    )
    baseline_summary.pop("target_outcomes")
    if targeted_summary["target_successes"] <= random_summary["target_successes"]:
        outcome = "negative"
        rationale = "Targeted success did not strictly exceed matched random success."
    elif regression_loss > regression_limit:
        outcome = "negative"
        rationale = "Targeted regression loss exceeded the locked limit."
    else:
        outcome = "positive"
        rationale = (
            "Targeted success strictly exceeded matched random success without exceeding the "
            "locked regression limit."
        )
    return {
        "gate": outcome,
        "rationale": rationale,
        "regression_limit": regression_limit,
        "targeted_regression_loss": regression_loss,
        "arms": summaries,
        "paired_targeted_vs_random": paired,
        "next_action": (
            "cancel_original_and_oracle_then_start_multitask_boundary_search"
            if outcome == "negative"
            else "release_original_then_assess_remaining_oracle_question"
        ),
    }

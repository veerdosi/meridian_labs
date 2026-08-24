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


def evaluate_dose_gate(
    *,
    baseline: list[dict],
    targeted: list[dict],
    random: list[dict],
    original: list[dict] | None,
    regression_limit: float,
    dose: int,
    prior_targeted: dict | None = None,
    expected_target: int = 40,
    expected_regression: int = 20,
) -> dict:
    """Apply the predeclared sequential comparison and marginal-dose rule."""
    arms = {
        "released_checkpoint": _arm_summary(
            baseline, expected_target=expected_target, expected_regression=expected_regression
        ),
        "targeted": _arm_summary(
            targeted, expected_target=expected_target, expected_regression=expected_regression
        ),
        "random": _arm_summary(
            random, expected_target=expected_target, expected_regression=expected_regression
        ),
    }
    if original is not None:
        arms["original_distribution"] = _arm_summary(
            original, expected_target=expected_target, expected_regression=expected_regression
        )
    baseline_summary, targeted_summary = arms["released_checkpoint"], arms["targeted"]
    regression_loss = baseline_summary["regression_success_rate"] - targeted_summary["regression_success_rate"]
    paired = {
        name: _paired_sign_test(targeted_summary["target_outcomes"], arms[name]["target_outcomes"])
        for name in ("random", "original_distribution")
        if name in arms
    }
    comparisons = {
        name: targeted_summary["target_successes"] > arms[name]["target_successes"]
        for name in ("random", "original_distribution")
        if name in arms
    }
    for summary in arms.values():
        summary.pop("target_outcomes")
    if regression_loss > regression_limit:
        decision, rationale = "stop", "targeted regression loss exceeded the locked limit"
    elif not comparisons["random"]:
        decision, rationale = "stop", "targeted did not strictly beat equal-dose random data"
    elif original is None:
        decision, rationale = "release_original", "targeted beat random; original-distribution comparison remains"
    elif not comparisons["original_distribution"]:
        decision, rationale = "stop", "targeted did not strictly beat equal-dose original-distribution data"
    elif prior_targeted is None:
        decision, rationale = "run_medium_dose", "small targeted dose beat both controls within the regression limit"
    else:
        marginal_successes = targeted_summary["target_successes"] - int(prior_targeted["target_successes"])
        if marginal_successes >= 2:
            decision, rationale = "select_medium_dose", "medium dose added at least 2/40 target successes and retained its advantage"
        else:
            decision, rationale = "select_small_dose", "medium dose added fewer than 2/40 target successes; the small dose is the stopping point"
    return {
        "schema": "meridian-physical-dose-gate-v1",
        "dose": dose,
        "decision": decision,
        "rationale": rationale,
        "regression_limit": regression_limit,
        "targeted_regression_loss": regression_loss,
        "strict_comparisons": comparisons,
        "paired": paired,
        "arms": arms,
    }

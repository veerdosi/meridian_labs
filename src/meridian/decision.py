from __future__ import annotations

from pydantic import BaseModel

from meridian.models import EvaluationResult, InterventionArm


class InterventionDecision(BaseModel):
    selected_arm: InterventionArm
    targeted_gain_over_none: float
    targeted_gain_over_random: float
    targeted_gain_over_original: float
    worst_regression: float
    total_su: float
    recommendation: str
    next_dose: int | None


def decide_intervention(
    evaluations: list[EvaluationResult],
    *,
    max_regression: float,
    current_dose: int,
    medium_dose: int,
) -> InterventionDecision:
    by_arm = {result.arm: result for result in evaluations}
    required = {
        InterventionArm.NONE,
        InterventionArm.TARGETED,
        InterventionArm.RANDOM,
        InterventionArm.ORIGINAL,
    }
    if missing := required - by_arm.keys():
        raise ValueError(f"missing comparison arms: {sorted(arm.value for arm in missing)}")
    none, targeted = by_arm[InterventionArm.NONE], by_arm[InterventionArm.TARGETED]
    random, original = by_arm[InterventionArm.RANDOM], by_arm[InterventionArm.ORIGINAL]
    gain_none = targeted.target_success_rate - none.target_success_rate
    gain_random = targeted.target_success_rate - random.target_success_rate
    gain_original = targeted.target_success_rate - original.target_success_rate
    worst_regression = min(result.regression_delta for result in evaluations)
    total_su = sum(result.cost.su or 0.0 for result in evaluations)
    if worst_regression < -max_regression:
        selected = InterventionArm.NONE
        recommendation = (
            "Reject the data intervention because its regression exceeds the locked limit."
        )
        next_dose = None
    elif gain_none <= 0 or gain_random <= 0 or gain_original <= 0:
        selected = InterventionArm.NONE
        recommendation = (
            "Do not collect more data; targeted data has not beaten all fair baselines."
        )
        next_dose = None
    elif current_dose < medium_dose and gain_random < 0.05:
        selected = InterventionArm.TARGETED
        recommendation = (
            "Evidence is positive but small; run the predeclared medium dose before scaling."
        )
        next_dose = medium_dose
    else:
        selected = InterventionArm.TARGETED
        recommendation = (
            "Targeted data is the highest-value tested intervention within the regression limit."
        )
        next_dose = None
    return InterventionDecision(
        selected_arm=selected,
        targeted_gain_over_none=gain_none,
        targeted_gain_over_random=gain_random,
        targeted_gain_over_original=gain_original,
        worst_regression=worst_regression,
        total_su=total_su,
        recommendation=recommendation,
        next_dose=next_dose,
    )

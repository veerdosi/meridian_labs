from meridian.decision import decide_intervention
from meridian.models import EvaluationResult, InterventionArm, ResourceCost


def result(arm: InterventionArm, target: float, regression: float = 0.0) -> EvaluationResult:
    return EvaluationResult(
        experiment_id="exp",
        intervention_id=f"int-{arm.value}",
        checkpoint_id=f"ckpt-{arm.value}",
        arm=arm,
        target_success_rate=target,
        target_ci=(max(0, target - 0.1), min(1, target + 0.1)),
        regression_success_rate=0.9 + regression,
        regression_delta=regression,
        seed_results={1: target},
        cost=ResourceCost(su=4, source="test"),
    )


def test_targeted_must_beat_fair_baselines() -> None:
    decision = decide_intervention(
        [
            result(InterventionArm.NONE, 0.2),
            result(InterventionArm.TARGETED, 0.4),
            result(InterventionArm.RANDOM, 0.35),
            result(InterventionArm.ORIGINAL, 0.3),
        ],
        max_regression=0.02,
        current_dose=24,
        medium_dose=64,
    )
    assert decision.selected_arm == InterventionArm.TARGETED
    assert decision.next_dose is None
    assert decision.total_su == 16


def test_regression_vetoes_gain() -> None:
    decision = decide_intervention(
        [
            result(InterventionArm.NONE, 0.2),
            result(InterventionArm.TARGETED, 0.7, regression=-0.03),
            result(InterventionArm.RANDOM, 0.3),
            result(InterventionArm.ORIGINAL, 0.3),
        ],
        max_regression=0.02,
        current_dose=24,
        medium_dose=64,
    )
    assert decision.selected_arm == InterventionArm.NONE


def test_unselected_baseline_regression_does_not_veto_targeted_arm() -> None:
    decision = decide_intervention(
        [
            result(InterventionArm.NONE, 0.2),
            result(InterventionArm.TARGETED, 0.7),
            result(InterventionArm.RANDOM, 0.3, regression=-0.03),
            result(InterventionArm.ORIGINAL, 0.3),
        ],
        max_regression=0.02,
        current_dose=24,
        medium_dose=64,
    )
    assert decision.selected_arm == InterventionArm.TARGETED
    assert decision.targeted_regression == 0.0
    assert decision.worst_regression == -0.03

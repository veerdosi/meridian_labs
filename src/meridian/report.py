from __future__ import annotations

from pathlib import Path

from meridian.models import CapabilityMap, EvaluationResult, ExperimentSpec, Hypothesis


def write_research_report(
    spec: ExperimentSpec,
    capability_map: CapabilityMap,
    hypotheses: list[Hypothesis],
    evaluations: list[EvaluationResult],
    output: Path,
) -> Path:
    ranked = sorted(evaluations, key=lambda result: result.target_success_rate, reverse=True)
    none = next(result for result in evaluations if result.arm.value == "none")
    targeted = next(result for result in evaluations if result.arm.value == "targeted")
    random = next(result for result in evaluations if result.arm.value == "random")
    original = next(result for result in evaluations if result.arm.value == "original_distribution")
    lines = [
        f"# Research result: {spec.id}",
        "",
        "## Boundary learned",
        "",
        (
            f"The adaptive search ran {capability_map.rollout_count} rollouts and observed "
            f"{capability_map.global_success_rate:.1%} overall success, with "
            f"{len(capability_map.failure_clusters)} failure regions and "
            f"{len(capability_map.boundary_pairs)} nearby success/failure pairs."
        ),
        "",
        "## Competing hypotheses",
        "",
    ]
    lines.extend(
        f"- **{h.kind.value}:** {h.statement} Test: {h.discriminating_test}" for h in hypotheses
    )
    lines.extend(
        [
            "",
            "## Controlled comparison",
            "",
            "| Arm | Target | 95% CI | Regression Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for result in ranked:
        lines.append(
            f"| {result.arm.value} | {result.target_success_rate:.1%} | "
            f"[{result.target_ci[0]:.1%}, {result.target_ci[1]:.1%}] | {result.regression_delta:+.1%} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"Targeted lift over no intervention: "
                f"{targeted.target_success_rate - none.target_success_rate:+.1%}. "
                f"Lift over matched random: "
                f"{targeted.target_success_rate - random.target_success_rate:+.1%}. "
                f"Lift over original-distribution data: "
                f"{targeted.target_success_rate - original.target_success_rate:+.1%}."
            ),
            "",
            (
                "This artifact reports uncertainty and regression explicitly. A surrogate result "
                "validates the harness contract only; it is not evidence about π0.5. The next run "
                "is the same locked contract against the released pi05_libero checkpoint on NSCC."
            ),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output

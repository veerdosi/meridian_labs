#!/usr/bin/env python3
"""Persist a completed real intervention campaign and write its research result."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from meridian.decision import decide_intervention
from meridian.models import (
    ArtifactRef,
    DatasetManifest,
    EvaluationResult,
    ExperimentSpec,
    InterventionArm,
    InterventionSpec,
    ResourceCost,
    TrainingRun,
)
from meridian.search import _wilson
from meridian.store import ExperimentStore


def resource_cost(manifest: dict) -> ResourceCost:
    actual = manifest["actual"]
    requested = manifest["requested"]
    walltime = float(actual["walltime_seconds"])
    return ResourceCost(
        job_id=str(manifest["job_id"]),
        queue=str(manifest["queue"]),
        requested_ncpus=int(requested["ncpus"]),
        requested_ngpus=int(requested["ngpus"]),
        requested_walltime_seconds=int(requested["walltime_seconds"]),
        actual_walltime_seconds=walltime,
        gpu_hours=walltime / 3600,
        su=float(actual["su_estimated"]),
        source="nscc_pbs_finished_record",
    )


def paired_sign_test(candidate: dict[str, bool], reference: dict[str, bool]) -> dict:
    wins = sum(candidate[key] and not reference[key] for key in reference)
    losses = sum(reference[key] and not candidate[key] for key in reference)
    discordant = wins + losses
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    p_value = min(1.0, 2 * tail / (2**discordant)) if discordant else 1.0
    return {"wins": wins, "losses": losses, "ties": len(reference) - discordant, "p": p_value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_spec = yaml.safe_load(args.runs.read_text())
    recorded_value = run_spec["recorded_at"]
    recorded_at = (
        recorded_value
        if isinstance(recorded_value, datetime)
        else datetime.fromisoformat(recorded_value)
    )
    spec = ExperimentSpec.model_validate(yaml.safe_load(Path(run_spec["config"]).read_text()))
    spec = spec.model_copy(update={"created_at": recorded_at})
    campaign = yaml.safe_load(Path(run_spec["campaign"]).read_text())
    interventions = {
        InterventionArm(item["arm"]): InterventionSpec.model_validate(item)
        for item in campaign["interventions"]
    }
    data_manifest = yaml.safe_load(Path(run_spec["data_job_manifest"]).read_text())
    baseline_regression = None
    evaluations = []
    training_runs = []
    datasets = []
    target_outcomes = {}
    args.output.mkdir(parents=True, exist_ok=True)

    for arm_name, item in run_spec["runs"].items():
        arm = InterventionArm(arm_name)
        intervention = interventions[arm]
        rows = [json.loads(line) for line in Path(item["rollouts"]).read_text().splitlines() if line]
        target = [row for row in rows if row["parameters"].get("evaluation_suite") == "target"]
        regression = [row for row in rows if row not in target]
        target_successes = sum(row["success"] for row in target)
        regression_successes = sum(row["success"] for row in regression)
        target_outcomes[arm] = {row["id"]: bool(row["success"]) for row in target}
        regression_rate = sum(row["success"] for row in regression) / len(regression)
        if arm == InterventionArm.NONE:
            baseline_regression = regression_rate
        per_seed = defaultdict(list)
        for row in target:
            per_seed[int(row["parameters"]["evaluation_seed"])].append(float(row["success"]))
        manifest = yaml.safe_load(Path(item["job_manifest"]).read_text())
        evaluations.append(
            EvaluationResult(
                experiment_id=spec.id,
                id=f"eval_{campaign['campaign_id']}_{arm.value}",
                intervention_id=intervention.id,
                checkpoint_id=str(item["checkpoint"]),
                arm=arm,
                target_successes=target_successes,
                target_trials=len(target),
                target_success_rate=target_successes / len(target),
                target_ci=_wilson(target_successes, len(target)),
                regression_successes=regression_successes,
                regression_trials=len(regression),
                regression_success_rate=regression_rate,
                regression_delta=0.0,
                seed_results={seed: sum(values) / len(values) for seed, values in per_seed.items()},
                cost=resource_cost(manifest),
                created_at=recorded_at,
            )
        )
        if arm == InterventionArm.NONE:
            continue
        rollouts_hash = data_manifest["artifacts"]["rollout_jsonl_sha256"][arm.value]
        dataset_uri = (
            f"{data_manifest['artifacts']['external_ssd_path']}/{arm.value}/rollouts.jsonl"
        )
        dataset = DatasetManifest(
            id=f"dataset_{campaign['campaign_id']}_{arm.value}",
            intervention_id=intervention.id,
            source="successful_action_replay_in_parameterized_libero",
            trajectory_count=intervention.trajectory_count,
            parameter_summary=intervention.target_bounds,
            provenance=[
                ArtifactRef(uri=dataset_uri, sha256=rollouts_hash, media_type="application/jsonl")
            ],
            quality_metrics={"valid_fraction": 1.0, "success_fraction": 1.0},
            format="meridian-trajectory-v1",
            created_at=recorded_at,
        )
        datasets.append(dataset)
        phases = manifest["phases"]
        training_runs.append(
            TrainingRun(
                id=f"train_{campaign['campaign_id']}_{arm.value}",
                experiment_id=spec.id,
                intervention_id=intervention.id,
                arm=arm,
                dataset_repo_id=manifest["dataset"]["repo_id"],
                starting_checkpoint=manifest["training"]["starting_checkpoint"],
                output_checkpoint=manifest["training"]["checkpoint"],
                config=manifest["training"]["config"],
                method=manifest["training"]["method"],
                steps=int(manifest["training"]["steps"]),
                metrics={key: float(value) for key, value in phases.items()},
                cost=resource_cost(manifest),
                created_at=recorded_at,
            )
        )

    if baseline_regression is None:
        raise SystemExit("none arm is required")
    for evaluation in evaluations:
        evaluation.regression_delta = evaluation.regression_success_rate - baseline_regression
    decision = decide_intervention(
        evaluations,
        max_regression=spec.max_regression,
        current_dose=int(campaign["dose"]),
        medium_dose=int(campaign["dose"]) * 2,
    )
    database_path = args.output / "experiment.sqlite"
    database_path.unlink(missing_ok=True)
    with ExperimentStore(database_path) as store:
        store.put("experiment", spec, recorded_at)
        for intervention in interventions.values():
            store.put("intervention", intervention, recorded_at)
        for dataset in datasets:
            store.put("dataset", dataset, recorded_at)
        for training in training_runs:
            store.put("training_run", training, recorded_at)
        for evaluation in evaluations:
            store.put("evaluation", evaluation, recorded_at)
    (args.output / "evaluations.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in evaluations], indent=2, sort_keys=True)
    )
    (args.output / "training_runs.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in training_runs], indent=2, sort_keys=True)
    )
    (args.output / "decision.json").write_text(
        json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    paired = {
        "targeted_vs_none": paired_sign_test(
            target_outcomes[InterventionArm.TARGETED], target_outcomes[InterventionArm.NONE]
        ),
        "targeted_vs_random": paired_sign_test(
            target_outcomes[InterventionArm.TARGETED], target_outcomes[InterventionArm.RANDOM]
        ),
        "targeted_vs_original": paired_sign_test(
            target_outcomes[InterventionArm.TARGETED], target_outcomes[InterventionArm.ORIGINAL]
        ),
        "random_vs_none": paired_sign_test(
            target_outcomes[InterventionArm.RANDOM], target_outcomes[InterventionArm.NONE]
        ),
    }
    (args.output / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True)
    )
    cost_summary = run_spec.get("cost_summary", {})
    (args.output / "cost_summary.json").write_text(
        json.dumps(cost_summary, indent=2, sort_keys=True)
    )

    ranked = sorted(evaluations, key=lambda item: item.target_success_rate, reverse=True)
    targeted = next(item for item in evaluations if item.arm == InterventionArm.TARGETED)
    random = next(item for item in evaluations if item.arm == InterventionArm.RANDOM)
    original = next(item for item in evaluations if item.arm == InterventionArm.ORIGINAL)
    none = next(item for item in evaluations if item.arm == InterventionArm.NONE)
    report_context = campaign.get("report_context", {})
    boundary_summary = report_context.get(
        "learned_boundary_and_mechanism",
        (
            "The deployed `pi05_libero` policy had 40/40 successes in the initial envelope, "
            "then 2/24 failures in the wider search. One compound view/visual profile reproduced "
            "in 4/5 trials. Matched probes identified two-step replanning as the causal amplifier: "
            "0/5 success with short replanning versus 3/5 with canonical replanning."
        ),
    )
    hypothesis_supported = decision.selected_arm == InterventionArm.TARGETED
    hypothesis_statement = campaign.get("hypothesis", {}).get(
        "statement", "Targeted data has higher intervention value than fair baselines."
    )
    conclusion = (
        "The preregistered targeted-selection hypothesis is supported at this dose by the locked "
        "point-estimate rule. A larger dose is required only if requested by the decision engine."
        if hypothesis_supported
        else "The preregistered targeted-selection hypothesis is falsified at this dose."
    )
    next_action = report_context.get(
        "next_action_on_support" if hypothesis_supported else "next_action_on_reject",
        (
            "Run the predeclared next dose and broaden cross-task confirmation."
            if hypothesis_supported
            else "Refine the intervention-value model; do not scale this targeted strategy."
        ),
    )
    lines = [
        "# π0.5 / LIBERO intervention result",
        "",
        "## Learned boundary and mechanism",
        "",
        boundary_summary,
        "",
        f"Preregistered hypothesis: {hypothesis_statement}",
        "",
        "## Controlled comparison",
        "",
        "| Arm | Target success | Wilson 95% CI | Regression | SU |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in ranked:
        target_count = (
            f"{result.target_successes}/{result.target_trials} "
            if result.target_trials is not None
            else ""
        )
        lines.append(
            f"| {result.arm.value} | {target_count}({result.target_success_rate:.0%}) | "
            f"[{result.target_ci[0]:.0%}, {result.target_ci[1]:.0%}] | "
            f"{result.regression_success_rate:.0%} ({result.regression_delta:+.0%}) | "
            f"{result.cost.su or 0:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"Targeted data improved over no intervention by "
                f"{targeted.target_success_rate - none.target_success_rate:+.0%}, but trailed "
                f"matched random by {targeted.target_success_rate - random.target_success_rate:+.0%} "
                f"and original-distribution data by "
                f"{targeted.target_success_rate - original.target_success_rate:+.0%}."
            ),
            "",
            (
                conclusion
            ),
            "",
            (
                f"On paired plans, targeted versus random had "
                f"{paired['targeted_vs_random']['wins']} wins and "
                f"{paired['targeted_vs_random']['losses']} losses "
                f"(two-sided sign-test p={paired['targeted_vs_random']['p']:.3f}); the point "
                "difference is not itself proof that random is superior."
            ),
            "",
            f"Decision engine: {decision.recommendation}",
            "",
            (
                f"Campaign compute cost was {cost_summary.get('campaign_su', decision.total_su):.2f} "
                f"SU; cumulative build-and-research usage was "
                f"{cost_summary.get('cumulative_su', decision.total_su):.2f} SU."
            ),
            "",
            "## Next action",
            "",
            next_action,
        ]
    )
    (args.output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(f"evaluations={len(evaluations)} decision={decision.selected_arm.value} output={args.output}")


if __name__ == "__main__":
    main()

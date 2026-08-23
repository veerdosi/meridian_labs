from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from meridian.adapters.base import PolicyAdapter
from meridian.models import (
    ArtifactRef,
    CapabilityMap,
    DatasetManifest,
    EvaluationResult,
    ExperimentSpec,
    Hypothesis,
    InterventionArm,
    InterventionSamplingStrategy,
    InterventionSpec,
    ResourceCost,
)
from meridian.search import _wilson


def sample_axis_value(
    *,
    strategy: InterventionSamplingStrategy,
    rng: np.random.Generator,
    axis_low: float,
    axis_high: float,
    target_low: float,
    target_high: float,
    canonical: float,
    target_fraction: float = 0.75,
    use_target_component: bool | None = None,
) -> float:
    """Sample one intervention axis using an explicit, auditable coverage strategy."""
    if strategy == InterventionSamplingStrategy.CANONICAL:
        return canonical
    if strategy == InterventionSamplingStrategy.CENTER:
        return (target_low + target_high) / 2
    if strategy == InterventionSamplingStrategy.FULL_UNIFORM:
        return float(rng.uniform(axis_low, axis_high))
    if strategy == InterventionSamplingStrategy.BOUNDS_UNIFORM:
        return float(rng.uniform(target_low, target_high))
    if strategy == InterventionSamplingStrategy.EVIDENCE_WEIGHTED_MIXTURE:
        target_component = (
            rng.random() < target_fraction
            if use_target_component is None
            else use_target_component
        )
        if not target_component:
            return float(rng.uniform(axis_low, axis_high))
        center = (target_low + target_high) / 2
        sigma = max((target_high - target_low) / 6, np.finfo(float).eps)
        return float(np.clip(rng.normal(center, sigma), target_low, target_high))
    raise ValueError(f"sampling strategy must be resolved before sampling: {strategy}")


def make_intervention_arms(
    spec: ExperimentSpec,
    hypothesis: Hypothesis,
    capability_map: CapabilityMap,
    dose: int,
    training_steps: int,
) -> list[InterventionSpec]:
    if not capability_map.failure_clusters:
        raise ValueError("cannot target an intervention without a failure region")
    target = capability_map.failure_clusters[0].bounds
    diversity = {axis.name: min(4, dose) for axis in spec.parameter_space.axes}
    common = {
        "experiment_id": spec.id,
        "hypothesis_id": hypothesis.id,
        "trajectory_count": dose,
        "target_bounds": target,
        "diversity_requirements": diversity,
        "quality_checks": [
            "finite_actions",
            "task_success",
            "parameter_coverage",
            "provenance_complete",
        ],
        "training_steps": training_steps,
    }
    return [
        InterventionSpec(
            arm=arm,
            seed=100 + index,
            **(
                {**common, "trajectory_count": 0, "training_steps": 0, "source": "none"}
                if arm == InterventionArm.NONE
                else common
            ),
        )
        for index, arm in enumerate(
            [
                InterventionArm.NONE,
                InterventionArm.TARGETED,
                InterventionArm.RANDOM,
                InterventionArm.ORIGINAL,
                InterventionArm.ORACLE,
            ]
        )
    ]


def materialize_surrogate_dataset(
    spec: ExperimentSpec, intervention: InterventionSpec, output: Path
) -> DatasetManifest:
    output.mkdir(parents=True, exist_ok=True)
    if intervention.trajectory_count == 0:
        points: list[dict[str, float]] = []
    else:
        rng = np.random.default_rng(intervention.seed)
        points = []
        for _ in range(intervention.trajectory_count):
            point = {}
            for axis in spec.parameter_space.axes:
                if intervention.arm in {InterventionArm.TARGETED, InterventionArm.ORACLE}:
                    low, high = intervention.target_bounds[axis.name]
                elif intervention.arm == InterventionArm.ORIGINAL:
                    center = (axis.low + axis.high) / 2
                    low, high = (
                        center - (axis.high - axis.low) * 0.15,
                        center + (axis.high - axis.low) * 0.15,
                    )
                else:
                    low, high = axis.low, axis.high
                point[axis.name] = float(rng.uniform(low, high))
            points.append(point)
    data = json.dumps(points, indent=2, sort_keys=True).encode()
    path = output / f"{intervention.id}.json"
    path.write_bytes(data)
    target_coverage = (
        1.0
        if intervention.arm == InterventionArm.ORACLE
        else 0.82
        if intervention.arm == InterventionArm.TARGETED
        else 0.22
        if intervention.arm == InterventionArm.RANDOM
        else 0.05
    )
    return DatasetManifest(
        intervention_id=intervention.id,
        source="parameterized_surrogate_simulation",
        trajectory_count=len(points),
        parameter_summary=intervention.target_bounds,
        provenance=[
            ArtifactRef(
                uri=str(path),
                sha256=hashlib.sha256(data).hexdigest(),
                media_type="application/json",
                size_bytes=len(data),
            )
        ],
        quality_metrics={"valid_fraction": 1.0, "target_coverage": target_coverage},
        format="meridian-trajectory-v1",
    )


def evaluate_checkpoint(
    adapter: PolicyAdapter,
    spec: ExperimentSpec,
    checkpoint_id: str,
    intervention: InterventionSpec,
    baseline_regression: float = 0.9,
) -> EvaluationResult:
    successes = []
    seed_results = {}
    for seed in spec.evaluation_seeds:
        per_seed = []
        rng = np.random.default_rng(seed)
        for index in range(20):
            parameters = {
                axis.name: float(rng.uniform(*intervention.target_bounds[axis.name]))
                for axis in spec.parameter_space.axes
            }
            per_seed.append(adapter.rollout(spec, parameters, seed * 1000 + index).success)
        seed_results[seed] = float(np.mean(per_seed))
        successes.extend(per_seed)
    low, high = _wilson(sum(successes), len(successes))
    regression = max(
        0.0, baseline_regression - (0.005 if intervention.arm == InterventionArm.ORACLE else 0.0)
    )
    return EvaluationResult(
        experiment_id=spec.id,
        intervention_id=intervention.id,
        checkpoint_id=checkpoint_id,
        arm=intervention.arm,
        target_success_rate=float(np.mean(successes)),
        target_ci=(low, high),
        regression_success_rate=regression,
        regression_delta=regression - baseline_regression,
        seed_results=seed_results,
        cost=ResourceCost(source="local_surrogate"),
    )

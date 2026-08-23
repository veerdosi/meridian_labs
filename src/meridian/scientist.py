from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from meridian.models import CapabilityMap, ExperimentSpec, Hypothesis, HypothesisKind, RolloutRecord


def evidence_package(
    spec: ExperimentSpec, capability_map: CapabilityMap, rollouts: list[RolloutRecord], output: Path
) -> Path:
    """Create the bounded machine-readable packet reviewed by the Codex scientist."""
    correlations = {}
    outcomes = np.asarray([float(r.success) for r in rollouts])
    for axis in spec.parameter_space.axes:
        values = np.asarray([r.parameters[axis.name] for r in rollouts])
        correlations[axis.name] = (
            float(np.corrcoef(values, outcomes)[0, 1]) if np.std(values) else 0.0
        )
    payload = {
        "contract_version": 1,
        "experiment": spec.model_dump(mode="json"),
        "capability_map": capability_map.model_dump(mode="json"),
        "axis_success_correlations": correlations,
        "required_output": {
            "hypotheses": "list[Hypothesis]",
            "selection_must_include_competing_non_data_explanation": True,
            "each_hypothesis_requires_discriminating_test_and_predicted_result": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output


def propose_competing_hypotheses(
    spec: ExperimentSpec, capability_map: CapabilityMap, rollouts: list[RolloutRecord]
) -> list[Hypothesis]:
    """Deterministic fallback; interactive Codex review can replace these records."""
    outcomes = np.asarray([float(r.success) for r in rollouts])
    correlations = {}
    for axis in spec.parameter_space.axes:
        values = np.asarray([r.parameters[axis.name] for r in rollouts])
        correlations[axis.name] = (
            float(np.corrcoef(values, outcomes)[0, 1]) if np.std(values) else 0.0
        )
    strongest = max(correlations, key=lambda name: abs(correlations[name]))
    evidence = (
        capability_map.failure_clusters[0].evidence_rollout_ids
        if capability_map.failure_clusters
        else []
    )
    return [
        Hypothesis(
            experiment_id=spec.id,
            kind=HypothesisKind.DATA_COVERAGE,
            statement=f"Training coverage is sparse at the observed extremes of {strongest}.",
            evidence_for=evidence,
            discriminating_test=f"Matched small-dose simulator trajectories concentrated in the failure-side {strongest} region.",
            predicted_result="Targeted data improves holdout boundary success more than equal-size random and original-distribution data.",
            priority=0.8,
        ),
        Hypothesis(
            experiment_id=spec.id,
            kind=HypothesisKind.CAMERA_TASK_CONFIG,
            statement="The boundary reflects observation geometry or camera configuration rather than missing demonstrations.",
            evidence_for=evidence,
            discriminating_test="Re-evaluate identical states after canonicalizing camera/task configuration without training.",
            predicted_result="Canonicalization recovers success immediately and added data has little marginal value.",
            priority=0.55,
        ),
        Hypothesis(
            experiment_id=spec.id,
            kind=HypothesisKind.INFERENCE_CONTROL,
            statement="Action chunking or control settings amplify errors near the discovered boundary.",
            evidence_for=[item for pair in capability_map.boundary_pairs[:10] for item in pair],
            discriminating_test="Sweep action horizon and replanning frequency on fixed paired boundary seeds.",
            predicted_result="A control-only setting change improves paired failures without training.",
            priority=0.4,
        ),
    ]

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from meridian.adapters.base import PolicyAdapter
from meridian.models import (
    DatasetManifest,
    EvaluationResult,
    ExperimentSpec,
    ResourceCost,
    RolloutRecord,
)


class BoundarySurrogateAdapter(PolicyAdapter):
    """Deterministic test subject with a non-axis-aligned hidden failure boundary."""

    def __init__(self) -> None:
        self.checkpoint_id = "unloaded"
        self.gain = 0.0

    def load(self, checkpoint_id: str, checkpoint_source: str) -> None:
        self.checkpoint_id = checkpoint_id

    def act(self, observation: dict[str, Any]) -> list[float]:
        return [0.0] * 7

    @staticmethod
    def probability(parameters: dict[str, float], gain: float = 0.0) -> float:
        camera = parameters.get("camera_yaw", 0.0)
        occlusion = parameters.get("occlusion", 0.0)
        pose = parameters.get("object_x", 0.0)
        difficulty = 5.2 * occlusion + 2.8 * abs(camera) + 2.0 * max(pose, 0.0)
        return 1.0 / (1.0 + math.exp(difficulty - 3.15 - gain))

    def rollout(
        self, spec: ExperimentSpec, parameters: dict[str, float], seed: int
    ) -> RolloutRecord:
        digest = hashlib.sha256(f"{seed}:{sorted(parameters.items())}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        p = self.probability(parameters, self.gain)
        return RolloutRecord(
            experiment_id=spec.id,
            checkpoint_id=self.checkpoint_id,
            task=spec.task_suite,
            seed=seed,
            parameters=parameters,
            success=draw < p,
            score=p,
            phase_labels=["grasp" if p < 0.65 else "complete"],
            steps=240 if draw < p else 180,
            duration_seconds=0.001,
            metadata={"surrogate": True},
        )

    def finetune(
        self,
        dataset: DatasetManifest,
        intervention_id: str,
        training_steps: int,
        seed: int,
        output: Path,
    ) -> tuple[str, ResourceCost]:
        output.mkdir(parents=True, exist_ok=True)
        quality = dataset.quality_metrics.get("target_coverage", 0.0)
        self.gain = 1.25 * quality * min(training_steps / 100.0, 1.0)
        checkpoint = f"surrogate-{intervention_id}-{seed}"
        return checkpoint, ResourceCost(
            requested_ncpus=1,
            actual_walltime_seconds=0.001,
            cpu_core_hours=0.000001,
            source="local_surrogate",
        )

    def evaluate(
        self, spec: ExperimentSpec, checkpoint_id: str, intervention_id: str
    ) -> EvaluationResult:
        raise NotImplementedError("evaluation is performed by the comparison runner")

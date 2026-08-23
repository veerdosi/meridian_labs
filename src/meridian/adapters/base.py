from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from meridian.models import (
    DatasetManifest,
    EvaluationResult,
    ExperimentSpec,
    ResourceCost,
    RolloutRecord,
)


class PolicyAdapter(ABC):
    """The only layer allowed to contain policy-family assumptions."""

    @abstractmethod
    def load(self, checkpoint_id: str, checkpoint_source: str) -> None: ...

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> list[float]: ...

    @abstractmethod
    def rollout(
        self, spec: ExperimentSpec, parameters: dict[str, float], seed: int
    ) -> RolloutRecord: ...

    @abstractmethod
    def finetune(
        self,
        dataset: DatasetManifest,
        intervention_id: str,
        training_steps: int,
        seed: int,
        output: Path,
    ) -> tuple[str, ResourceCost]: ...

    @abstractmethod
    def evaluate(
        self, spec: ExperimentSpec, checkpoint_id: str, intervention_id: str
    ) -> EvaluationResult: ...

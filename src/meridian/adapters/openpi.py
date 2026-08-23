from __future__ import annotations

from collections.abc import Callable
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

RolloutExecutor = Callable[[ExperimentSpec, dict[str, float], int, str], RolloutRecord]
Trainer = Callable[[DatasetManifest, str, int, int, Path], tuple[str, ResourceCost]]
Evaluator = Callable[[ExperimentSpec, str, str], EvaluationResult]


class OpenPILiberoAdapter(PolicyAdapter):
    """Thin OpenPI adapter; expensive operations are injected scheduler executors."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        rollout_executor: RolloutExecutor | None = None,
        trainer: Trainer | None = None,
        evaluator: Evaluator | None = None,
        config_name: str = "pi05_libero",
    ) -> None:
        self.host = host
        self.port = port
        self.rollout_executor = rollout_executor
        self.trainer = trainer
        self.evaluator = evaluator
        self.config_name = config_name
        self.checkpoint_id = "unloaded"
        self.checkpoint_source = ""
        self._client: Any = None

    def load(self, checkpoint_id: str, checkpoint_source: str) -> None:
        self.checkpoint_id = checkpoint_id
        self.checkpoint_source = checkpoint_source

    def act(self, observation: dict[str, Any]) -> list[float]:
        if self._client is None:
            try:
                from openpi_client.websocket_client_policy import WebsocketClientPolicy
            except ImportError as error:
                raise RuntimeError("install openpi-client to use live inference") from error
            self._client = WebsocketClientPolicy(self.host, self.port)
        response = self._client.infer(observation)
        return response["actions"]

    def rollout(
        self, spec: ExperimentSpec, parameters: dict[str, float], seed: int
    ) -> RolloutRecord:
        if self.rollout_executor is None:
            raise RuntimeError("rollouts require a bounded scheduler executor")
        return self.rollout_executor(spec, parameters, seed, self.checkpoint_id)

    def finetune(
        self,
        dataset: DatasetManifest,
        intervention_id: str,
        training_steps: int,
        seed: int,
        output: Path,
    ) -> tuple[str, ResourceCost]:
        if self.trainer is None:
            raise RuntimeError("fine-tuning requires a bounded scheduler trainer")
        return self.trainer(dataset, intervention_id, training_steps, seed, output)

    def evaluate(
        self, spec: ExperimentSpec, checkpoint_id: str, intervention_id: str
    ) -> EvaluationResult:
        if self.evaluator is None:
            raise RuntimeError("evaluation requires a bounded scheduler evaluator")
        return self.evaluator(spec, checkpoint_id, intervention_id)

    def server_command(self, python: Path, openpi_root: Path, *, record: bool = True) -> list[str]:
        command = [
            str(python),
            str(openpi_root / "scripts/serve_policy.py"),
            "policy:checkpoint",
            f"--policy.config={self.config_name}",
            f"--policy.dir={self.checkpoint_source}",
            f"--port={self.port}",
        ]
        if record:
            command.append("--record")
        return command

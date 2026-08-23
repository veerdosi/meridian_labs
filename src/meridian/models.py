from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class ArtifactRef(BaseModel):
    uri: str
    sha256: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None


class ParameterAxis(BaseModel):
    name: str
    low: float
    high: float
    semantic: str

    @model_validator(mode="after")
    def valid_bounds(self) -> ParameterAxis:
        if self.high <= self.low:
            raise ValueError("axis high must be greater than low")
        return self


class ParameterSpace(BaseModel):
    axes: list[ParameterAxis]

    @model_validator(mode="after")
    def unique_axes(self) -> ParameterSpace:
        names = [axis.name for axis in self.axes]
        if not names or len(names) != len(set(names)):
            raise ValueError("parameter space requires unique axes")
        return self


class Budget(BaseModel):
    max_rollouts: int = Field(gt=0)
    max_training_variants: int = Field(default=4, gt=0)
    max_gpu_hours: float | None = Field(default=None, gt=0)
    max_su: float | None = Field(default=None, gt=0)


class ExperimentSpec(BaseModel):
    id: str = Field(default_factory=lambda: new_id("exp"))
    created_at: datetime = Field(default_factory=utc_now)
    policy_adapter: str
    checkpoint_id: str
    checkpoint_source: str
    task_suite: str
    target_metric: str = "success"
    regression_suites: list[str]
    evaluation_seeds: list[int]
    parameter_space: ParameterSpace
    budget: Budget
    max_regression: float = Field(default=0.02, ge=0, le=1)


class RolloutRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rollout"))
    experiment_id: str
    checkpoint_id: str
    task: str
    seed: int
    parameters: dict[str, float]
    success: bool
    score: float = Field(ge=0, le=1)
    phase_labels: list[str] = Field(default_factory=list)
    steps: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RegionSummary(BaseModel):
    bounds: dict[str, tuple[float, float]]
    count: int
    successes: int
    success_rate: float
    wilson_low: float
    wilson_high: float
    evidence_rollout_ids: list[str]


class CapabilityMap(BaseModel):
    id: str = Field(default_factory=lambda: new_id("map"))
    experiment_id: str
    rollout_count: int
    global_success_rate: float
    regions: list[RegionSummary]
    failure_clusters: list[RegionSummary]
    boundary_pairs: list[tuple[str, str]]
    created_at: datetime = Field(default_factory=utc_now)


class HypothesisKind(StrEnum):
    DATA_COVERAGE = "data_coverage"
    ADAPTER = "adapter"
    NORMALIZATION = "normalization"
    CAMERA_TASK_CONFIG = "camera_task_config"
    INFERENCE_CONTROL = "inference_control"
    TRAINING_CONFIG = "training_config"


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: new_id("hyp"))
    experiment_id: str
    kind: HypothesisKind
    statement: str
    evidence_for: list[str]
    evidence_against: list[str] = Field(default_factory=list)
    discriminating_test: str
    predicted_result: str
    priority: float = Field(ge=0, le=1)
    status: Literal["proposed", "selected", "rejected", "supported"] = "proposed"


class InterventionArm(StrEnum):
    NONE = "none"
    TARGETED = "targeted"
    RANDOM = "random"
    ORIGINAL = "original_distribution"
    ORACLE = "oracle_targeted"
    NO_DATA_FIX = "no_data_fix"


class InterventionSpec(BaseModel):
    id: str = Field(default_factory=lambda: new_id("intervention"))
    experiment_id: str
    hypothesis_id: str
    arm: InterventionArm
    source: Literal["simulation", "none"] = "simulation"
    trajectory_count: int = Field(ge=0)
    target_bounds: dict[str, tuple[float, float]]
    diversity_requirements: dict[str, int]
    quality_checks: list[str]
    expected_gain: float | None = None
    training_steps: int = Field(default=0, ge=0)
    seed: int


class DatasetManifest(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dataset"))
    intervention_id: str
    source: str
    trajectory_count: int
    parameter_summary: dict[str, tuple[float, float]]
    provenance: list[ArtifactRef]
    quality_metrics: dict[str, float]
    format: str
    created_at: datetime = Field(default_factory=utc_now)


class ResourceCost(BaseModel):
    job_id: str | None = None
    queue: str | None = None
    requested_ncpus: int = Field(default=0, ge=0)
    requested_ngpus: int = Field(default=0, ge=0)
    requested_memory_gb: float = Field(default=0, ge=0)
    requested_walltime_seconds: int = Field(default=0, ge=0)
    actual_walltime_seconds: float = Field(default=0, ge=0)
    cpu_core_hours: float = Field(default=0, ge=0)
    gpu_hours: float = Field(default=0, ge=0)
    su: float | None = Field(default=None, ge=0)
    source: str


class EvaluationResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("eval"))
    experiment_id: str
    intervention_id: str
    checkpoint_id: str
    arm: InterventionArm
    target_success_rate: float = Field(ge=0, le=1)
    target_ci: tuple[float, float]
    regression_success_rate: float = Field(ge=0, le=1)
    regression_delta: float
    seed_results: dict[int, float]
    cost: ResourceCost
    created_at: datetime = Field(default_factory=utc_now)

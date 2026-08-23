import json
from pathlib import Path

from meridian.adapters.surrogate import BoundarySurrogateAdapter
from meridian.models import (
    Budget,
    ExperimentSpec,
    InterventionArm,
    ParameterAxis,
    ParameterSpace,
    ResourceCost,
    TrainingRun,
)
from meridian.scientist import evidence_package
from meridian.search import AdaptiveFailureSearch, build_capability_map, propose_parameter_points
from meridian.store import ExperimentStore


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        policy_adapter="surrogate",
        checkpoint_id="test",
        checkpoint_source="generated",
        task_suite="test",
        regression_suites=["central"],
        evaluation_seeds=[1, 2],
        budget=Budget(max_rollouts=36),
        parameter_space=ParameterSpace(
            axes=[
                ParameterAxis(name="camera_yaw", low=-1, high=1, semantic="viewpoint"),
                ParameterAxis(name="occlusion", low=0, high=1, semantic="occlusion"),
                ParameterAxis(name="object_x", low=-1, high=1, semantic="pose"),
            ]
        ),
    )


def test_store_round_trip(tmp_path: Path) -> None:
    value = spec()
    with ExperimentStore(tmp_path / "store.sqlite") as store:
        store.put("experiment", value)
        assert store.get("experiment", value.id, ExperimentSpec) == value

        training = TrainingRun(
            experiment_id=value.id,
            intervention_id="targeted",
            arm=InterventionArm.TARGETED,
            dataset_repo_id="meridian/test",
            starting_checkpoint="released",
            output_checkpoint="checkpoint/99",
            config="meridian_pi05_libero",
            method="LoRA",
            steps=100,
            cost=ResourceCost(source="pbs", su=1.0),
        )
        store.put("training_run", training)
        assert store.get("training_run", training.id, TrainingRun) == training


def test_adaptive_search_builds_boundary_map(tmp_path: Path) -> None:
    value = spec()
    adapter = BoundarySurrogateAdapter()
    adapter.load(value.checkpoint_id, value.checkpoint_source)
    with ExperimentStore(tmp_path / "store.sqlite") as store:
        rollouts = AdaptiveFailureSearch(value, adapter, store).run(seed=4)
    capability_map = build_capability_map(value, rollouts)
    assert len(rollouts) == value.budget.max_rollouts
    assert any(r.success for r in rollouts)
    assert any(not r.success for r in rollouts)
    assert capability_map.failure_clusters
    assert capability_map.boundary_pairs
    proposals = propose_parameter_points(value, rollouts, count=4, seed=9)
    assert len(proposals) == 4
    assert all(set(point) == {"camera_yaw", "occlusion", "object_x"} for point in proposals)


def test_evidence_correlations_are_json_safe_for_constant_outcomes(tmp_path: Path) -> None:
    value = spec()
    adapter = BoundarySurrogateAdapter()
    adapter.load(value.checkpoint_id, value.checkpoint_source)
    rollouts = [
        adapter.rollout(value, {"camera_yaw": x, "occlusion": 0.0, "object_x": 0.0}, seed=1)
        for x in (-0.1, 0.0, 0.1)
    ]
    assert all(record.success for record in rollouts)
    output = evidence_package(value, build_capability_map(value, rollouts), rollouts, tmp_path / "evidence.json")
    payload = json.loads(output.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    assert set(payload["axis_success_correlations"].values()) == {0.0}

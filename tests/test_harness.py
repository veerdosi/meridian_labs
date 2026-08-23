from pathlib import Path

from meridian.adapters.surrogate import BoundarySurrogateAdapter
from meridian.models import Budget, ExperimentSpec, ParameterAxis, ParameterSpace
from meridian.search import AdaptiveFailureSearch, build_capability_map
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

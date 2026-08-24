import pytest

from meridian.data_selection import select_data_arms


def record(
    identifier: str,
    position: float,
    *,
    init_state: int = 0,
    phase: str = "screening",
    suite: str = "libero_object",
    task: int = 0,
) -> dict:
    return {
        "id": identifier,
        "success": True,
        "task_suite": suite,
        "task_id": task,
        "init_state_index": init_state,
        "parameters": {"phase": phase},
        "initial_free_joint_positions": {"object": [position, 0.0, 0.0]},
    }


def test_data_arms_are_unique_nested_and_targeted_by_geometry() -> None:
    config = {
        "task_set": [{"suite": "libero_object", "task_id": 0}],
        "partitions": {
            "screening_init_states": [0, 1, 2, 3],
            "training_init_states": [4, 5, 6, 7, 8, 9],
        },
        "training": {"doses": [2, 4], "source_seed": 5},
        "thresholds": {"minimum_unique_targeted_sources": 4},
    }
    selection = {
        "selected": {
            "task_suite": "libero_object",
            "task_id": 0,
            "coverage_rule": {
                "feature": "object:object:x",
                "threshold": 0.5,
                "failure_side": "low",
            },
        }
    }
    boundary = record("failure", 0.0)
    sources = [record(f"screen{i}", 5.0 + i, init_state=i) for i in range(4)] + [
        record(
            f"source{i}",
            float(i),
            init_state=4 + i,
            phase="training_source_pool",
        )
        for i in range(6)
    ]
    result = select_data_arms(selection, boundary, sources, config)
    assert result["doses"]["2"]["targeted"] == ["source0", "source1"]
    assert result["doses"]["4"]["targeted"][:2] == result["doses"]["2"]["targeted"]
    assert len(set(result["doses"]["4"]["random"])) == 4


def test_data_selection_rejects_duplicate_padding() -> None:
    config = {
        "task_set": [{"suite": "libero_object", "task_id": 0}],
        "partitions": {"screening_init_states": [0], "training_init_states": [1, 2, 3, 4]},
        "training": {"doses": [4], "source_seed": 5},
        "thresholds": {"minimum_unique_targeted_sources": 4},
    }
    universe = [record("same", 4.0, init_state=0)] + [
        record(
            "same" if index == 1 else f"source{index}",
            float(index),
            init_state=index,
            phase="training_source_pool",
        )
        for index in (1, 2, 3, 4)
    ]
    with pytest.raises(ValueError, match="duplicate rollout IDs"):
        select_data_arms(
            {
                "selected": {
                    "task_suite": "libero_object",
                    "task_id": 0,
                    "coverage_rule": {
                        "feature": "object:object:x",
                        "threshold": 0.5,
                        "failure_side": "low",
                    },
                }
            },
            record("failure", 0.0),
            universe,
            config,
        )

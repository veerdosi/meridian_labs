import pytest

from meridian.data_selection import select_data_arms


def record(identifier: str, position: float, *, suite: str = "libero_object", task: int = 0) -> dict:
    return {"id": identifier, "success": True, "task_suite": suite, "task_id": task, "initial_free_joint_positions": {"object": [position, 0.0, 0.0]}}


def test_data_arms_are_unique_nested_and_targeted_by_geometry() -> None:
    config = {"training": {"doses": [2, 4], "source_seed": 5}, "thresholds": {"minimum_unique_targeted_sources": 4}}
    selection = {"selected": {"task_suite": "libero_object", "task_id": 0}}
    boundary = record("failure", 0.0)
    sources = [record(f"s{i}", float(i)) for i in range(6)] + [record("other", 9.0, suite="libero_goal")]
    result = select_data_arms(selection, boundary, sources, config)
    assert result["doses"]["2"]["targeted"] == ["s0", "s1"]
    assert result["doses"]["4"]["targeted"][:2] == result["doses"]["2"]["targeted"]
    assert len(set(result["doses"]["4"]["random"])) == 4
    assert result["training_authorized"] is False


def test_data_selection_rejects_duplicate_padding() -> None:
    config = {"training": {"doses": [4], "source_seed": 5}, "thresholds": {"minimum_unique_targeted_sources": 4}}
    with pytest.raises(ValueError, match="unique successful"):
        select_data_arms(
            {"selected": {"task_suite": "libero_object", "task_id": 0}},
            record("failure", 0.0),
            [record("same", 0.1), record("same", 0.2)],
            config,
        )

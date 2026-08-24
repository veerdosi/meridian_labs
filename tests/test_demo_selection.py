import pytest

from meridian.demo_selection import select_partitioned_maximin


def test_demo_selection_excludes_holdout_and_deduplicates_nearest_states() -> None:
    inventory = {index: [float(index), 0.0] for index in range(6)}
    demos = {
        "demo_a": [1.01, 0.0],
        "demo_b": [1.02, 0.0],
        "demo_c": [2.01, 0.0],
        "demo_d": [3.01, 0.0],
        "demo_holdout": [5.0, 0.0],
    }
    selected = select_partitioned_maximin(
        demos,
        inventory,
        allowed_indices=[1, 2, 3],
        forbidden_indices=[4, 5],
        count=3,
    )
    assert {item["demo"] for item in selected} == {"demo_a", "demo_c", "demo_d"}
    assert {item["nearest_init_state_index"] for item in selected} == {1, 2, 3}


def test_demo_selection_rejects_overlapping_partitions() -> None:
    with pytest.raises(ValueError, match="overlap"):
        select_partitioned_maximin(
            {"demo": [0.0, 1.0]},
            {0: [0.0, 1.0], 1: [1.0, 1.0]},
            allowed_indices=[0],
            forbidden_indices=[0],
            count=1,
        )

import pytest

from scripts.inventory_libero_initial_states import validate_inventory_config


def config() -> dict:
    return {
        "task_set": [{"suite": "libero_goal", "task_id": 0}],
        "partitions": {
            "screening_init_states": [0],
            "confirmation_init_states": [1],
            "training_init_states": [2],
            "untouched_holdout_init_states": [3],
        },
    }


def test_inventory_config_accepts_unique_disjoint_states() -> None:
    validate_inventory_config(config())


def test_inventory_config_rejects_partition_overlap() -> None:
    value = config()
    value["partitions"]["untouched_holdout_init_states"] = [2]
    with pytest.raises(ValueError, match="overlap"):
        validate_inventory_config(value)

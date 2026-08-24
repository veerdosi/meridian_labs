from pathlib import Path

import numpy as np
import pytest

from meridian.rollout_integrity import (
    aligned_trajectory_arrays,
    canonical_sha256,
    pad_contact_pairs,
    reserve_results_path,
    validate_plans,
)


def valid_plan(**updates: object) -> dict:
    plan = {"id": "p0", "task_suite": "libero_object", "task_id": 1, "seed": 2}
    plan.update(updates)
    return plan


def test_plan_validation_rejects_duplicates_and_legacy_interventions() -> None:
    with pytest.raises(ValueError, match="duplicate plan id"):
        validate_plans([valid_plan(), valid_plan()])
    with pytest.raises(ValueError, match="intervention fields"):
        validate_plans([valid_plan(camera_x=0.1)])
    assert validate_plans([valid_plan(camera_x=0.0)]) == [valid_plan(camera_x=0.0)]


def test_hash_is_independent_of_mapping_order() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_trajectory_alignment_and_contact_padding() -> None:
    arrays = aligned_trajectory_arrays(
        {"image": np.zeros((2, 1)), "actions": np.ones((2, 1))}, ("image", "actions")
    )
    assert len(arrays) == 2
    padded = pad_contact_pairs(
        [np.asarray([[1, 2]]), np.asarray([[3, 4], [5, 6]]), np.empty((0, 2))]
    )
    assert padded.shape == (3, 2, 2)
    assert padded[0, 1].tolist() == [-1, -1]


def test_results_path_reservation_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "run" / "rollouts.jsonl"
    reserve_results_path(path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reserve_results_path(path)

import hashlib
from pathlib import Path

import numpy as np
import pytest

from meridian.rollout_integrity import (
    aligned_trajectory_arrays,
    canonical_sha256,
    pad_contact_pairs,
    reserve_results_path,
    validate_plans,
    verify_physical_rollout,
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
    with pytest.raises(ValueError, match="intervention fields"):
        validate_plans([valid_plan(camera_x=0.0)])
    with pytest.raises(ValueError, match="unknown fields"):
        validate_plans([valid_plan(typo_field=1)])


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


def test_physical_preflight_requires_aligned_predicates_features_and_videos(
    tmp_path: Path,
) -> None:
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    trace = tmp_path / "trajectory.npz"
    np.savez_compressed(
        trace,
        image=frames,
        clean_observer_image=frames,
        policy_image=frames,
        wrist_image=frames,
        state=np.zeros((2, 8)),
        actions=np.zeros((2, 7)),
        sim_qpos=np.zeros((2, 3)),
        sim_qvel=np.zeros((2, 3)),
        contact_count=np.zeros(2),
        contact_geom_ids=np.full((2, 1, 2), -1),
        goal_predicate_satisfied_before=np.zeros((2, 1), dtype=bool),
        goal_predicate_satisfied_after=np.zeros((2, 1), dtype=bool),
        goal_argument_positions_after=np.zeros((2, 2, 3)),
    )
    videos = {}
    for name in ("clean_observer", "policy_input", "wrist", "diagnostic"):
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(b"video")
        videos[name] = str(path)
    record = {
        "id": "physical-telemetry-preflight",
        "trace": str(trace),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "simulator_schema": {
            "nq": 3,
            "goal_predicates": [["In", "object", "basket"]],
            "goal_arguments": ["basket", "object"],
        },
        "initial_sim_qpos": [0, 0, 0],
        "initial_physical_features": {"object:cube:x": 0.0},
        "videos": videos,
        "steps": 2,
    }
    assert verify_physical_rollout(record)["verified"] is True

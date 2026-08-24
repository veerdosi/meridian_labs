import hashlib
from pathlib import Path

import numpy as np
import pytest

from meridian.dataset_integrity import validate_training_records


def write_trace(path: Path, *, finite: bool = True, action_value: float = 0.0) -> str:
    state = np.zeros((2, 8), dtype=np.float32)
    if not finite:
        state[0, 0] = np.nan
    np.savez_compressed(
        path,
        image=np.zeros((2, 256, 256, 3), dtype=np.uint8),
        wrist_image=np.zeros((2, 256, 256, 3), dtype=np.uint8),
        state=state,
        actions=np.full((2, 7), action_value, dtype=np.float32),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(identifier: str, trace: Path, digest: str) -> dict:
    return {"id": identifier, "trace": str(trace), "trace_sha256": digest, "success": True}


def test_training_dataset_requires_exact_unique_verified_episodes(tmp_path: Path) -> None:
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    records = [
        record("a", first, write_trace(first)),
        record("b", second, write_trace(second, action_value=0.1)),
    ]
    assert len(validate_training_records(records, expected_episodes=2)) == 2
    with pytest.raises(ValueError, match="expected 3"):
        validate_training_records(records, expected_episodes=3)
    with pytest.raises(ValueError, match="duplicate rollout IDs"):
        validate_training_records([records[0], {**records[1], "id": "a"}], expected_episodes=2)


def test_training_dataset_rejects_hash_mismatch_and_nonfinite_arrays(tmp_path: Path) -> None:
    bad_hash = tmp_path / "hash.npz"
    write_trace(bad_hash)
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_training_records([record("a", bad_hash, "0" * 64)], expected_episodes=1)
    nonfinite = tmp_path / "nonfinite.npz"
    with pytest.raises(ValueError, match="non-finite"):
        validate_training_records(
            [record("b", nonfinite, write_trace(nonfinite, finite=False))], expected_episodes=1
        )

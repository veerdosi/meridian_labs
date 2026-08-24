import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.replay_source_contract import load_replay_sources, sha256_file

ROOT = Path(__file__).parents[1]
REAL_RECORD = ROOT / "tests/fixtures/viewpoint-v1-training-source-000.json"


def test_contract_resolves_relocated_archive_and_rejects_unreferenced_failures(
    tmp_path: Path,
) -> None:
    episode_id = "source-real-schema"
    trace_root = tmp_path / "trace-root"
    episode = trace_root / "episodes" / episode_id
    episode.mkdir(parents=True)
    trace = episode / "trajectory.npz"
    np.savez_compressed(
        trace,
        image=np.zeros((2, 4, 5, 3), dtype=np.uint8),
        wrist_image=np.zeros((2, 2, 3, 3), dtype=np.uint8),
        state=np.zeros((2, 8), dtype=np.float32),
        actions=np.zeros((2, 7), dtype=np.float32),
    )
    source = {
        "id": episode_id,
        "success": True,
        "task_suite": "libero_10",
        "task_id": 3,
        "seed": 1,
        "init_state_index": 0,
        "trace": "/purged/job-specific/path/trajectory.npz",
        "trace_sha256": sha256_file(trace),
    }
    unrelated_failure = {**source, "id": "failed-target", "success": False}
    results = tmp_path / "rollouts.jsonl"
    results.write_text(json.dumps(source) + "\n" + json.dumps(unrelated_failure) + "\n")
    loaded = load_replay_sources(results, required_ids={episode_id}, trace_root=trace_root)
    assert loaded[episode_id]["trace"] == str(trace.resolve())


def test_contract_rejects_failed_referenced_source(tmp_path: Path) -> None:
    source = {
        "id": "failed-source",
        "success": False,
        "task_suite": "libero_10",
        "task_id": 3,
        "seed": 1,
        "init_state_index": 0,
        "trace": "/missing/trajectory.npz",
        "trace_sha256": "0" * 64,
    }
    results = tmp_path / "rollouts.jsonl"
    results.write_text(json.dumps(source) + "\n")
    with pytest.raises(ValueError, match="not successful"):
        load_replay_sources(results, required_ids={"failed-source"})


@pytest.mark.integration
def test_actual_evaluation_record_and_trajectory_archive(tmp_path: Path) -> None:
    fixture_root = os.environ.get("MERIDIAN_REAL_REPLAY_FIXTURE_ROOT")
    if fixture_root is None:
        pytest.skip("set MERIDIAN_REAL_REPLAY_FIXTURE_ROOT to the hydrated actual archive")
    record = json.loads(REAL_RECORD.read_text())
    results = tmp_path / "rollouts.jsonl"
    results.write_text(json.dumps(record) + "\n")
    loaded = load_replay_sources(
        results,
        required_ids={record["id"]},
        trace_root=Path(fixture_root),
    )
    assert loaded[record["id"]]["trace_sha256"] == record["trace_sha256"]

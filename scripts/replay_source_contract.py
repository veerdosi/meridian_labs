"""Resolve and validate successful cross-job trajectory sources for replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REQUIRED_RESULT_FIELDS = {
    "id",
    "success",
    "task_suite",
    "task_id",
    "seed",
    "init_state_index",
    "trace",
    "trace_sha256",
}
REQUIRED_ARCHIVE_KEYS = {"image", "wrist_image", "state", "actions"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_trace(record: dict, *, results_path: Path, trace_root: Path | None) -> Path:
    trace = Path(str(record["trace"]))
    candidates = [trace]
    if trace_root is not None:
        candidates.extend(
            [
                trace_root / "episodes" / str(record["id"]) / "trajectory.npz",
                trace_root / str(record["id"]) / "trajectory.npz",
            ]
        )
    candidates.append(results_path.parent / "episodes" / str(record["id"]) / "trajectory.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"trajectory for {record['id']} was not found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def validate_archive(path: Path, record: dict) -> None:
    expected_hash = str(record["trace_sha256"])
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"trajectory hash mismatch for {record['id']}: {actual_hash} != {expected_hash}"
        )
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARCHIVE_KEYS - set(archive.files)
        if missing:
            raise ValueError(f"trajectory {record['id']} is missing keys: {sorted(missing)}")
        lengths = {key: len(archive[key]) for key in REQUIRED_ARCHIVE_KEYS}
        if not lengths["actions"] or len(set(lengths.values())) != 1:
            raise ValueError(f"trajectory {record['id']} has inconsistent lengths: {lengths}")
        if archive["actions"].ndim != 2 or archive["state"].ndim != 2:
            raise ValueError(f"trajectory {record['id']} has invalid action/state rank")
        if archive["image"].ndim != 4 or archive["wrist_image"].ndim != 4:
            raise ValueError(f"trajectory {record['id']} has invalid image rank")


def load_replay_sources(
    results_path: Path,
    *,
    required_ids: set[str] | None = None,
    trace_root: Path | None = None,
) -> dict[str, dict]:
    records = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    by_id = {}
    for record in records:
        missing = REQUIRED_RESULT_FIELDS - record.keys()
        if missing:
            raise ValueError(f"evaluation record is missing fields: {sorted(missing)}")
        if record["id"] in by_id:
            raise ValueError(f"duplicate evaluation record: {record['id']}")
        by_id[record["id"]] = record
    selected_ids = required_ids if required_ids is not None else {
        record_id for record_id, record in by_id.items() if record["success"]
    }
    missing_ids = selected_ids - by_id.keys()
    if missing_ids:
        raise ValueError(f"replay plans reference unknown sources: {sorted(missing_ids)}")
    selected = {}
    for record_id in sorted(selected_ids):
        record = by_id[record_id]
        if not record["success"]:
            raise ValueError(f"replay source is not successful: {record_id}")
        resolved_trace = resolve_trace(record, results_path=results_path, trace_root=trace_root)
        validate_archive(resolved_trace, record)
        selected[record_id] = {**record, "trace": str(resolved_trace)}
    if not selected:
        raise ValueError("visual intervention replay requires at least one successful source rollout")
    return selected

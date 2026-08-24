"""Fail-closed validation for real intervention trajectory datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from meridian.rollout_integrity import aligned_trajectory_arrays, file_sha256


def validate_training_records(
    records: Sequence[Mapping[str, Any]], *, expected_episodes: int
) -> list[dict[str, Any]]:
    if len(records) != expected_episodes:
        raise ValueError(f"dataset has {len(records)} episodes; expected {expected_episodes}")
    identifiers = [str(record["id"]) for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset contains duplicate rollout IDs")
    traces = [str(record["trace"]) for record in records]
    if len(traces) != len(set(traces)):
        raise ValueError("dataset contains duplicate trajectory paths")
    verified = []
    hashes = set()
    for record in records:
        if not bool(record.get("success")):
            raise ValueError(f"dataset record {record['id']} is not successful")
        expected_hash = record.get("trace_sha256")
        if not expected_hash:
            raise ValueError(f"dataset record {record['id']} has no trace hash")
        trace = Path(str(record["trace"]))
        observed_hash = file_sha256(trace)
        if observed_hash != expected_hash:
            raise ValueError(
                f"trace hash mismatch for {record['id']}: expected {expected_hash}, observed {observed_hash}"
            )
        if observed_hash in hashes:
            raise ValueError("dataset contains duplicate trajectory content")
        hashes.add(observed_hash)
        with np.load(trace) as trajectory:
            images, wrists, states, actions = aligned_trajectory_arrays(
                trajectory, ("image", "wrist_image", "state", "actions")
            )
            if images.shape[1:] != (256, 256, 3) or wrists.shape[1:] != (256, 256, 3):
                raise ValueError(f"record {record['id']} has invalid image shapes")
            if states.shape[1:] != (8,) or actions.shape[1:] != (7,):
                raise ValueError(f"record {record['id']} has invalid state/action shapes")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"record {record['id']} has non-finite state or action values")
        verified.append(dict(record))
    return verified

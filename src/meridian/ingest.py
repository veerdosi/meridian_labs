from __future__ import annotations

import json
from pathlib import Path

from meridian.models import ArtifactRef, ExperimentSpec, RolloutRecord


def ingest_libero_rollouts(
    spec: ExperimentSpec, path: Path, checkpoint_id: str | None = None
) -> list[RolloutRecord]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        parameters = {
            key: float(raw["parameters"][key])
            for key in (axis.name for axis in spec.parameter_space.axes)
        }
        artifacts = [
            ArtifactRef(
                uri=raw["trace"], sha256=raw.get("trace_sha256"), media_type="application/x-npz"
            )
        ]
        if raw.get("video"):
            artifacts.append(ArtifactRef(uri=raw["video"], media_type="video/mp4"))
        records.append(
            RolloutRecord(
                id=raw["id"],
                experiment_id=spec.id,
                checkpoint_id=checkpoint_id or spec.checkpoint_id,
                task=raw["task"],
                seed=raw["seed"],
                parameters=parameters,
                success=raw["success"],
                score=raw["score"],
                phase_labels=raw.get("phase_labels", []),
                steps=raw["steps"],
                duration_seconds=raw["duration_seconds"],
                artifacts=artifacts,
                metadata={
                    "task_suite": raw["task_suite"],
                    "task_id": raw["task_id"],
                    "init_state_index": raw["init_state_index"],
                    "inference_seconds": raw.get("inference_seconds"),
                },
            )
        )
    return records

"""Integrity and telemetry contracts shared by simulator rollout entry points."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_PLAN_FIELDS = {"id", "task_suite", "task_id", "seed"}
FORBIDDEN_INTERVENTION_FIELDS = {
    "action_noise",
    "brightness",
    "camera_x",
    "camera_yaw_deg",
    "object_joint",
    "object_x",
    "object_y",
    "occlusion",
    "visual_distractors",
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plans(plans: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not plans:
        raise ValueError("plan file contains no plans")
    validated: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(plans):
        plan = dict(raw)
        missing = REQUIRED_PLAN_FIELDS - plan.keys()
        if missing:
            raise ValueError(f"plan {index} missing required fields: {sorted(missing)}")
        identifier = str(plan["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate plan id: {identifier}")
        identifiers.add(identifier)
        unsupported = {
            key
            for key in FORBIDDEN_INTERVENTION_FIELDS
            if key in plan and plan[key] not in (None, 0, 0.0, "")
        }
        if unsupported:
            raise ValueError(
                "legacy or unvalidated intervention fields are forbidden: "
                f"{sorted(unsupported)}"
            )
        if int(plan["task_id"]) < 0 or int(plan["seed"]) < 0:
            raise ValueError(f"plan {identifier} has a negative task_id or seed")
        if int(plan.get("init_state_index", 0)) < 0:
            raise ValueError(f"plan {identifier} has a negative init_state_index")
        if int(plan.get("replan_steps", 5)) < 1:
            raise ValueError(f"plan {identifier} has replan_steps < 1")
        validated.append(plan)
    return validated


def aligned_trajectory_arrays(
    trajectory: Mapping[str, np.ndarray], keys: Iterable[str]
) -> tuple[np.ndarray, ...]:
    names = tuple(keys)
    missing = [name for name in names if name not in trajectory]
    if missing:
        raise ValueError(f"trajectory missing arrays: {missing}")
    arrays = tuple(np.asarray(trajectory[name]) for name in names)
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        observed = {name: len(array) for name, array in zip(names, arrays)}
        raise ValueError(f"trajectory arrays are not aligned: {observed}")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("trajectory contains no frames")
    return arrays


def simulator_state_sha256(qpos: np.ndarray, qvel: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)):
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def simulator_metadata(sim: Any) -> dict[str, Any]:
    model = sim.model
    return {
        "joint_names": [model.joint_id2name(index) for index in range(int(model.njnt))],
        "joint_types": [int(value) for value in np.asarray(model.jnt_type)],
        "joint_qpos_addresses": [int(value) for value in np.asarray(model.jnt_qposadr)],
        "body_names": [model.body_id2name(index) for index in range(int(model.nbody))],
        "geom_names": [model.geom_id2name(index) for index in range(int(model.ngeom))],
        "nq": int(model.nq),
        "nv": int(model.nv),
    }


def contact_pairs(sim: Any) -> np.ndarray:
    """Return the active MuJoCo contact geom IDs for one synchronized frame."""
    return np.asarray(
        [[int(sim.data.contact[index].geom1), int(sim.data.contact[index].geom2)]
         for index in range(int(sim.data.ncon))],
        dtype=np.int32,
    ).reshape(-1, 2)


def pad_contact_pairs(steps: Sequence[np.ndarray]) -> np.ndarray:
    """Encode variable contact sets as a pickle-free T x C x 2 integer array."""
    if not steps:
        return np.empty((0, 0, 2), dtype=np.int32)
    maximum = max(len(np.asarray(step)) for step in steps)
    padded = np.full((len(steps), maximum, 2), -1, dtype=np.int32)
    for index, step in enumerate(steps):
        pairs = np.asarray(step, dtype=np.int32).reshape(-1, 2)
        padded[index, : len(pairs)] = pairs
    return padded


def reserve_results_path(path: Path) -> None:
    """Fail closed instead of silently appending duplicate scientific outcomes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x"):
            pass
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite or append existing results: {path}") from error

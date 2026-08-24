"""Integrity and telemetry contracts shared by simulator rollout entry points."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_PLAN_FIELDS = {"id", "task_suite", "task_id", "seed"}
ALLOWED_PLAN_FIELDS = REQUIRED_PLAN_FIELDS | {
    "evaluation_suite",
    "init_state_index",
    "max_steps",
    "phase",
    "repeat",
    "replan_steps",
    "wait_steps",
}
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
        unsupported = {key for key in FORBIDDEN_INTERVENTION_FIELDS if key in plan}
        if unsupported:
            raise ValueError(
                "legacy or unvalidated intervention fields are forbidden: "
                f"{sorted(unsupported)}"
            )
        if unknown := plan.keys() - ALLOWED_PLAN_FIELDS:
            raise ValueError(f"plan {index} has unknown fields: {sorted(unknown)}")
        identifier = str(plan["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate plan id: {identifier}")
        identifiers.add(identifier)
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


def free_joint_positions(sim: Any) -> dict[str, list[float]]:
    """Extract initial xyz positions for physical objects represented by free joints."""
    model = sim.model
    positions: dict[str, list[float]] = {}
    for joint_id, joint_type in enumerate(np.asarray(model.jnt_type)):
        if int(joint_type) != 0:  # MuJoCo mjJNT_FREE
            continue
        address = int(model.jnt_qposadr[joint_id])
        name = model.joint_id2name(joint_id) or f"joint_{joint_id}"
        positions[str(name)] = [float(value) for value in sim.data.qpos[address : address + 3]]
    return positions


def initial_physical_features(sim: Any) -> dict[str, float]:
    """Expose interpretable initial object, articulation, and robot configuration axes."""
    model = sim.model
    features: dict[str, float] = {}
    robot_tokens = ("robot", "panda", "gripper", "finger", "hand")
    for joint_id, joint_type_value in enumerate(np.asarray(model.jnt_type)):
        joint_type = int(joint_type_value)
        address = int(model.jnt_qposadr[joint_id])
        name = str(model.joint_id2name(joint_id) or f"joint_{joint_id}")
        lowered = name.lower()
        if joint_type == 0:
            for offset, axis in enumerate(("x", "y", "z")):
                features[f"object:{name}:{axis}"] = float(sim.data.qpos[address + offset])
        elif joint_type in (2, 3):
            category = "robot" if any(token in lowered for token in robot_tokens) else "articulation"
            features[f"{category}:{name}"] = float(sim.data.qpos[address])
    return features


def goal_metadata(env: Any) -> dict[str, Any]:
    """Return LIBERO's exact BDDL goal predicates and ordered argument names."""
    goals = [list(state) for state in env.env.parsed_problem["goal_state"]]
    arguments = sorted({str(argument) for state in goals for argument in state[1:]})
    return {"predicates": goals, "arguments": arguments}


def evaluate_goal_predicates(env: Any, predicates: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray([bool(env.env._eval_predicate(list(state))) for state in predicates], dtype=bool)


def goal_argument_positions(env: Any, arguments: Sequence[str]) -> np.ndarray:
    positions = []
    for argument in arguments:
        state = env.env.object_states_dict[str(argument)]
        positions.append(np.asarray(state.get_geom_state()["pos"], dtype=np.float64))
    return np.asarray(positions, dtype=np.float64).reshape(-1, 3)


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


def verify_physical_rollout(record: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one real natural-state rollout before any campaign is allowed to start."""
    required_record_fields = {
        "trace",
        "trace_sha256",
        "simulator_schema",
        "initial_sim_qpos",
        "initial_physical_features",
        "videos",
    }
    if missing := required_record_fields - record.keys():
        raise ValueError(f"rollout record missing physical fields: {sorted(missing)}")
    trace = Path(str(record["trace"]))
    if file_sha256(trace) != record["trace_sha256"]:
        raise ValueError("physical rollout trace hash mismatch")
    temporal_keys = (
        "image",
        "clean_observer_image",
        "policy_image",
        "wrist_image",
        "state",
        "actions",
        "sim_qpos",
        "sim_qvel",
        "contact_count",
        "contact_geom_ids",
        "goal_predicate_satisfied_before",
        "goal_predicate_satisfied_after",
        "goal_argument_positions_after",
    )
    with np.load(trace) as trajectory:
        arrays = aligned_trajectory_arrays(trajectory, temporal_keys)
        by_name = dict(zip(temporal_keys, arrays))
        if not np.array_equal(by_name["image"], by_name["policy_image"]):
            raise ValueError("image and exact policy input differ under the canonical preflight")
        if not np.array_equal(by_name["clean_observer_image"], by_name["policy_image"]):
            raise ValueError("clean and policy streams differ under the canonical preflight")
        if float(np.std(by_name["policy_image"])) < 1.0:
            raise ValueError("policy input is visually degenerate")
        for name in ("state", "actions", "sim_qpos", "sim_qvel", "goal_argument_positions_after"):
            if not np.isfinite(by_name[name]).all():
                raise ValueError(f"physical telemetry contains non-finite values in {name}")
        schema = record["simulator_schema"]
        predicate_count = len(schema.get("goal_predicates", []))
        argument_count = len(schema.get("goal_arguments", []))
        if predicate_count == 0:
            raise ValueError("rollout exposes no BDDL goal predicates")
        if by_name["goal_predicate_satisfied_after"].shape[1:] != (predicate_count,):
            raise ValueError("goal predicate telemetry does not match its schema")
        if by_name["goal_predicate_satisfied_before"].shape[1:] != (predicate_count,):
            raise ValueError("pre-action goal predicate telemetry does not match its schema")
        if by_name["goal_argument_positions_after"].shape[1:] != (argument_count, 3):
            raise ValueError("goal argument telemetry does not match its schema")
        if by_name["sim_qpos"].shape[1:] != (int(schema["nq"]),):
            raise ValueError("qpos telemetry does not match its simulator schema")
    if len(record["initial_sim_qpos"]) != int(record["simulator_schema"]["nq"]):
        raise ValueError("initial qpos does not match its simulator schema")
    if not record["initial_physical_features"]:
        raise ValueError("rollout exposes no interpretable physical initial-state features")
    expected_videos = {"clean_observer", "policy_input", "wrist", "diagnostic"}
    if set(record["videos"]) != expected_videos:
        raise ValueError("rollout does not reference exactly four evidence videos")
    for path in record["videos"].values():
        video = Path(str(path))
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"missing or empty evidence video: {video}")
    return {
        "schema": "meridian-physical-preflight-verification-v1",
        "id": record["id"],
        "steps": int(record["steps"]),
        "goal_predicates": len(record["simulator_schema"]["goal_predicates"]),
        "physical_features": len(record["initial_physical_features"]),
        "videos": 4,
        "verified": True,
    }

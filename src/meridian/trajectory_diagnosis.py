"""Evidence-linked stage summaries for physical LIBERO trajectories.

These summaries localize observable failure stages. They do not infer a unique root cause or
declare that data is the answer; that requires cross-rollout comparisons and control probes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROBOT_GEOM_TOKENS = ("robot", "panda", "gripper", "finger", "hand")


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _free_joint_motion(qpos: np.ndarray, schema: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, kind, address in zip(
        schema["joint_names"], schema["joint_types"], schema["joint_qpos_addresses"]
    ):
        if int(kind) != 0:  # MuJoCo mjJNT_FREE
            continue
        positions = qpos[:, int(address) : int(address) + 3]
        displacement = np.linalg.norm(positions - positions[0], axis=1)
        peak_index = int(np.argmax(displacement))
        result[str(name)] = {
            "maximum_translation_m": float(displacement[peak_index]),
            "final_translation_m": float(displacement[-1]),
            "peak_step_fraction": peak_index / max(1, len(displacement) - 1),
        }
    return result


def _articulation_motion(qpos: np.ndarray, schema: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name, kind, address in zip(
        schema["joint_names"], schema["joint_types"], schema["joint_qpos_addresses"]
    ):
        lowered = str(name).lower()
        if int(kind) not in (2, 3) or any(token in lowered for token in ROBOT_GEOM_TOKENS):
            continue
        values = qpos[:, int(address)]
        result[str(name)] = float(np.max(np.abs(values - values[0])))
    return result


def _robot_contact_mask(contact_ids: np.ndarray, schema: Mapping[str, Any]) -> np.ndarray:
    names = list(schema["geom_names"])
    mask = np.zeros(len(contact_ids), dtype=bool)
    for step, pairs in enumerate(contact_ids):
        for left, right in np.asarray(pairs).reshape(-1, 2):
            if int(left) < 0 or int(right) < 0:
                continue
            pair_names = (names[int(left)] or "", names[int(right)] or "")
            robot_sides = [
                any(token in name.lower() for token in ROBOT_GEOM_TOKENS)
                for name in pair_names
            ]
            if robot_sides[0] != robot_sides[1]:
                mask[step] = True
                break
    return mask


def trajectory_metrics(trajectory: Mapping[str, np.ndarray], schema: Mapping[str, Any]) -> dict:
    states = np.asarray(trajectory["state"])
    qpos = np.asarray(trajectory["sim_qpos"])
    contacts = np.asarray(trajectory["contact_geom_ids"])
    if not (len(states) == len(qpos) == len(contacts)) or len(states) == 0:
        raise ValueError("state, sim_qpos, and contact telemetry must be non-empty and aligned")
    motions = _free_joint_motion(qpos, schema)
    articulations = _articulation_motion(qpos, schema)
    moving_joint = max(
        motions,
        key=lambda name: motions[name]["maximum_translation_m"],
        default=None,
    )
    goal_target_objects = list(
        dict.fromkeys(
            str(predicate[1])
            for predicate in schema.get("goal_predicates", [])
            if len(predicate) > 1
        )
    )
    target_joints = [
        name
        for name in motions
        if any(
            name == target or name.startswith(f"{target}_joint")
            for target in goal_target_objects
        )
    ]
    target_joint = max(
        target_joints,
        key=lambda name: motions[name]["maximum_translation_m"],
        default=None,
    )
    distractor_joints = [name for name in motions if name not in target_joints]
    distractor_joint = max(
        distractor_joints,
        key=lambda name: motions[name]["maximum_translation_m"],
        default=None,
    )
    robot_contact = _robot_contact_mask(contacts, schema)
    contact_steps = np.flatnonzero(robot_contact)
    predicate_count = len(schema.get("goal_predicates", []))
    predicate_after = np.asarray(
        trajectory["goal_predicate_satisfied_after"]
        if "goal_predicate_satisfied_after" in trajectory
        else np.zeros((len(states), predicate_count), dtype=bool),
        dtype=bool,
    )
    if len(predicate_after) != len(states):
        raise ValueError("goal predicate telemetry is not aligned")
    ever = predicate_after.any(axis=0) if predicate_count else np.zeros(0, dtype=bool)
    final = predicate_after[-1] if predicate_count else np.zeros(0, dtype=bool)
    return {
        "steps": len(states),
        "eef_path_m": _path_length(states[:, :3]),
        "gripper_range": float(np.ptp(states[:, 6:])) if states.shape[1] > 6 else 0.0,
        "robot_contact_fraction": float(robot_contact.mean()),
        "first_robot_contact_step": int(contact_steps[0]) if len(contact_steps) else None,
        "most_moved_free_joint": moving_joint,
        "free_joint_motion": motions,
        "goal_target_objects": goal_target_objects,
        "goal_target_free_joint": target_joint,
        "max_target_object_translation_m": (
            motions[target_joint]["maximum_translation_m"] if target_joint else None
        ),
        "final_target_object_translation_m": (
            motions[target_joint]["final_translation_m"] if target_joint else None
        ),
        "most_moved_distractor_free_joint": distractor_joint,
        "max_distractor_translation_m": (
            motions[distractor_joint]["maximum_translation_m"] if distractor_joint else 0.0
        ),
        "articulated_joint_motion": articulations,
        "max_articulation_change": max(articulations.values(), default=0.0),
        "max_object_translation_m": (
            motions[moving_joint]["maximum_translation_m"] if moving_joint else 0.0
        ),
        "final_object_translation_m": (
            motions[moving_joint]["final_translation_m"] if moving_joint else 0.0
        ),
        "object_peak_step_fraction": (
            motions[moving_joint]["peak_step_fraction"] if moving_joint else 0.0
        ),
        "goal_predicate_count": predicate_count,
        "goal_predicates_ever_satisfied": int(ever.sum()),
        "goal_predicates_finally_satisfied": int(final.sum()),
        "goal_predicate_regressions": int(np.logical_and(ever, ~final).sum()),
        "goal_predicate_labels": [list(value) for value in schema.get("goal_predicates", [])],
    }


def diagnose_stage(
    metrics: Mapping[str, Any],
    *,
    success: bool,
    min_eef_path_m: float = 0.05,
    min_object_motion_m: float = 0.025,
    retained_motion_fraction: float = 0.5,
    min_articulation_change: float = 0.10,
) -> dict:
    """Map physical observables to a stage hypothesis and explicit alternatives."""
    if success:
        return {
            "stage": "complete",
            "confidence": "observed",
            "evidence": ["environment success predicate was reached"],
            "candidate_missing_data": [],
            "alternatives": [],
        }
    eef_path = float(metrics["eef_path_m"])
    contact_fraction = float(metrics["robot_contact_fraction"])
    peak_motion = float(metrics["max_object_translation_m"])
    final_motion = float(metrics["final_object_translation_m"])
    articulation = float(metrics.get("max_articulation_change", 0.0))
    target_peak_value = metrics.get("max_target_object_translation_m")
    target_peak = float(target_peak_value) if target_peak_value is not None else None
    distractor_peak = float(metrics.get("max_distractor_translation_m", 0.0))
    physical_progress = peak_motion >= min_object_motion_m or articulation >= min_articulation_change
    evidence = [
        f"end-effector path={eef_path:.4f} m",
        f"robot-contact fraction={contact_fraction:.3f}",
        f"maximum object translation={peak_motion:.4f} m",
        f"final object translation={final_motion:.4f} m",
        f"maximum non-robot articulation change={articulation:.4f}",
        (
            "goal-target translation=unavailable"
            if target_peak is None
            else f"maximum goal-target translation={target_peak:.4f} m"
        ),
        f"maximum distractor translation={distractor_peak:.4f} m",
        (
            "goal predicates finally satisfied="
            f"{int(metrics.get('goal_predicates_finally_satisfied', 0))}/"
            f"{int(metrics.get('goal_predicate_count', 0))}"
        ),
    ]
    predicate_count = int(metrics.get("goal_predicate_count", 0))
    predicate_final = int(metrics.get("goal_predicates_finally_satisfied", 0))
    predicate_regressions = int(metrics.get("goal_predicate_regressions", 0))
    if predicate_regressions:
        stage = "recovery_or_subgoal_regression"
        missing = ["recovery", "temporal", "contact_state"]
        alternatives = ["later actions undid a completed subgoal", "unstable control"]
    elif predicate_count > 1 and 0 < predicate_final < predicate_count:
        stage = "sequencing"
        missing = ["temporal", "strategy", "sequencing"]
        alternatives = ["insufficient horizon", "second-subgoal localization failure"]
    elif (
        target_peak is not None
        and target_peak < min_object_motion_m
        and distractor_peak >= min_object_motion_m
    ):
        stage = "object_selection_or_role_binding"
        missing = ["task", "object", "strategy"]
        alternatives = [
            "language-role grounding failure",
            "policy reused a familiar but incorrect task strategy",
        ]
    elif eef_path < min_eef_path_m:
        stage = "localization_or_strategy"
        missing = ["initial_state", "object_pose", "strategy"]
        alternatives = ["policy-server/action-adapter fault", "language grounding failure"]
    elif contact_fraction == 0 and not physical_progress:
        stage = "approach"
        missing = ["initial_state", "object_pose", "approach"]
        alternatives = ["localization failure", "insufficient horizon"]
    elif not physical_progress:
        stage = "contact_or_grasp_control"
        missing = ["contact_state", "object_pose", "strategy"]
        alternatives = ["action scaling or replanning mismatch", "wrong contacted body"]
    elif final_motion < retained_motion_fraction * peak_motion:
        stage = "control_or_recovery"
        missing = ["contact_state", "recovery", "temporal"]
        alternatives = ["unstable action scaling", "goal-state mislocalization"]
    else:
        stage = "sequencing_or_termination"
        missing = ["temporal", "strategy", "termination"]
        alternatives = ["goal predicate not satisfied", "insufficient horizon"]
    return {
        "stage": stage,
        "confidence": "stage-level hypothesis; causal ambiguity remains",
        "evidence": evidence,
        "candidate_missing_data": missing,
        "alternatives": alternatives,
    }


def diagnose_record(record: Mapping[str, Any]) -> dict:
    trace = Path(str(record["trace"]))
    with np.load(trace) as trajectory:
        metrics = trajectory_metrics(trajectory, record["simulator_schema"])
    return {
        "id": record["id"],
        "success": bool(record["success"]),
        "metrics": metrics,
        "diagnosis": diagnose_stage(metrics, success=bool(record["success"])),
        "evidence_trace": str(trace),
        "trace_sha256": record.get("trace_sha256"),
    }

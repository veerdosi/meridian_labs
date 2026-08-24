import numpy as np

from meridian.trajectory_diagnosis import diagnose_stage, trajectory_metrics

SCHEMA = {
    "joint_names": ["cube_free"],
    "joint_types": [0],
    "joint_qpos_addresses": [0],
    "geom_names": ["robot0_rightfinger", "cube", "table"],
    "goal_predicates": [["On", "cube", "table"]],
}


def trajectory(*, eef_distance: float, object_peak: float, contact: bool, recover: bool = False):
    steps = 5
    state = np.zeros((steps, 8))
    state[:, 0] = np.linspace(0, eef_distance, steps)
    qpos = np.zeros((steps, 7))
    motion = np.linspace(0, object_peak, steps)
    if recover:
        motion[-1] = 0
    qpos[:, 0] = motion
    pairs = np.full((steps, 1, 2), -1, dtype=np.int32)
    if contact:
        pairs[:, 0] = [0, 1]
    return {"state": state, "sim_qpos": qpos, "contact_geom_ids": pairs}


def test_metrics_link_motion_and_robot_contact() -> None:
    metrics = trajectory_metrics(
        trajectory(eef_distance=0.2, object_peak=0.1, contact=True), SCHEMA
    )
    assert metrics["most_moved_free_joint"] == "cube_free"
    assert metrics["max_object_translation_m"] == 0.1
    assert metrics["robot_contact_fraction"] == 1.0


def test_stage_diagnosis_distinguishes_observable_failure_phases() -> None:
    approach = trajectory_metrics(
        trajectory(eef_distance=0.2, object_peak=0.0, contact=False), SCHEMA
    )
    contact = trajectory_metrics(
        trajectory(eef_distance=0.2, object_peak=0.0, contact=True), SCHEMA
    )
    recovery = trajectory_metrics(
        trajectory(eef_distance=0.2, object_peak=0.1, contact=True, recover=True), SCHEMA
    )
    assert diagnose_stage(approach, success=False)["stage"] == "approach"
    assert diagnose_stage(contact, success=False)["stage"] == "contact_or_grasp_control"
    assert diagnose_stage(recovery, success=False)["stage"] == "control_or_recovery"


def test_success_is_not_reinterpreted() -> None:
    assert diagnose_stage({}, success=True)["stage"] == "complete"


def test_exact_bddl_subgoal_progress_takes_precedence() -> None:
    metrics = {
        "eef_path_m": 0.2,
        "robot_contact_fraction": 0.2,
        "max_object_translation_m": 0.1,
        "final_object_translation_m": 0.1,
        "goal_predicate_count": 2,
        "goal_predicates_finally_satisfied": 1,
        "goal_predicate_regressions": 0,
    }
    assert diagnose_stage(metrics, success=False)["stage"] == "sequencing"


def test_wrong_object_motion_is_role_binding_not_progress() -> None:
    schema = {
        "joint_names": ["target_joint0", "distractor_joint0"],
        "joint_types": [0, 0],
        "joint_qpos_addresses": [0, 7],
        "geom_names": ["robot0_rightfinger", "target", "distractor"],
        "goal_predicates": [["on", "target", "region"]],
    }
    steps = 5
    qpos = np.zeros((steps, 14))
    qpos[:, 7] = np.linspace(0, 0.2, steps)
    trajectory_value = {
        "state": np.column_stack((np.linspace(0, 0.2, steps), np.zeros((steps, 7)))),
        "sim_qpos": qpos,
        "contact_geom_ids": np.full((steps, 1, 2), -1, dtype=np.int32),
    }
    metrics = trajectory_metrics(trajectory_value, schema)
    assert metrics["goal_target_free_joint"] == "target_joint0"
    assert metrics["most_moved_distractor_free_joint"] == "distractor_joint0"
    assert metrics["max_target_object_translation_m"] == 0.0
    assert metrics["max_distractor_translation_m"] == 0.2
    assert diagnose_stage(metrics, success=False)["stage"] == "object_selection_or_role_binding"

"""Deterministic gate for a repeated target-versus-distractor failure mechanism."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def assess_role_confirmation(
    rollouts: Sequence[Mapping[str, Any]],
    diagnoses: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict:
    gate = config["confirmation_gate"]
    threshold = float(gate["object_motion_threshold_m"])
    diagnosis_by_id = {str(item["id"]): item for item in diagnoses}
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rollout in rollouts:
        identifier = str(rollout["id"])
        if identifier not in diagnosis_by_id:
            raise ValueError(f"missing diagnosis for {identifier}")
        metrics = diagnosis_by_id[identifier]["metrics"]
        target_motion = metrics.get("max_target_object_translation_m")
        distractor_motion = float(metrics.get("max_distractor_translation_m", 0.0))
        wrong_object = (
            target_motion is not None
            and float(target_motion) < threshold
            and distractor_motion >= threshold
        )
        phase = str(rollout.get("parameters", {}).get("phase", ""))
        grouped[(str(rollout["task_suite"]), int(rollout["task_id"]))].append(
            {
                "id": identifier,
                "success": bool(rollout["success"]),
                "phase": phase,
                "wrong_object": wrong_object,
                "target_motion_m": target_motion,
                "distractor_motion_m": distractor_motion,
            }
        )
    expected_tasks = {
        (str(task["suite"]), int(task["task_id"])) for task in config["tasks"]
    }
    if set(grouped) != expected_tasks:
        raise ValueError("confirmation results do not match the locked task set")

    expected_per_task = int(config["confirmation"]["trials"]) // len(expected_tasks)
    task_summaries = []
    pooled_wrong = 0
    for suite, task_id in sorted(grouped):
        records = grouped[(suite, task_id)]
        if len(records) != expected_per_task:
            raise ValueError(
                f"{suite} task {task_id} has {len(records)}/{expected_per_task} trials"
            )
        successes = sum(item["success"] for item in records)
        wrong_count = sum(item["wrong_object"] for item in records)
        pooled_wrong += wrong_count
        extended_successes = sum(
            item["success"] for item in records if item["phase"].endswith("extended_horizon")
        )
        replanning_successes = sum(
            item["success"]
            for item in records
            if item["phase"].endswith(("rapid_replan", "long_chunk"))
        )
        checks = {
            "overall_success": successes / len(records)
            <= float(gate["require_each_task_overall_success_at_most"]),
            "wrong_object_fraction": wrong_count / len(records)
            >= float(gate["require_each_task_wrong_object_fraction_at_least"]),
            "extended_horizon": extended_successes
            == int(gate["require_each_task_extended_horizon_successes"]),
            "replanning_controls": replanning_successes
            <= int(gate["require_each_task_replanning_control_successes_at_most"]),
        }
        task_summaries.append(
            {
                "task_suite": suite,
                "task_id": task_id,
                "trials": len(records),
                "successes": successes,
                "wrong_object_trials": wrong_count,
                "extended_horizon_successes": extended_successes,
                "replanning_control_successes": replanning_successes,
                "checks": checks,
                "pass": all(checks.values()),
                "records": records,
            }
        )
    pooled_check = pooled_wrong >= int(gate["require_pooled_wrong_object_trials_at_least"])
    automated_pass = pooled_check and all(item["pass"] for item in task_summaries)
    return {
        "schema": "meridian-task-role-confirmation-v1",
        "automated_pass": automated_pass,
        "requires_human_review": bool(gate["human_review_required"]),
        "decision": "requires_human_review" if automated_pass else "reject_boundary",
        "pooled_wrong_object_trials": pooled_wrong,
        "pooled_wrong_object_check": pooled_check,
        "tasks": task_summaries,
    }


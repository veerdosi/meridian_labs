#!/usr/bin/env python3
"""Verify simulator-expert output before it is eligible for policy training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from meridian.rollout_integrity import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "validation", "training"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    plans = [json.loads(line) for line in args.plans.read_text().splitlines() if line.strip()]
    records = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    expected_ids = [str(item["id"]) for item in plans]
    observed_ids = [str(item["id"]) for item in records]
    if observed_ids != expected_ids:
        raise ValueError("expert results do not match the ordered locked plan")
    if len({item["state_spec_sha256"] for item in plans}) != len(plans):
        raise ValueError("expert plan repeats a physical state specification")

    grouped: dict[int, list[dict]] = defaultdict(list)
    verified = []
    for plan, record in zip(plans, records):
        if not bool(record.get("accepted")):
            grouped[int(plan["task_id"])].append(record)
            continue
        trace = Path(str(record["trace"]))
        if file_sha256(trace) != record["trace_sha256"]:
            raise ValueError(f"trace hash mismatch: {trace}")
        with np.load(trace) as trajectory:
            keys = (
                "clean_observer_image",
                "policy_image",
                "wrist_image",
                "state",
                "actions",
                "sim_qpos",
                "sim_qvel",
                "contact_count",
                "goal_predicate_satisfied_before",
                "goal_predicate_satisfied_after",
                "commanded_object_position_after",
                "other_object_position_after",
                "stage",
            )
            arrays = {key: np.asarray(trajectory[key]) for key in keys}
            lengths = {len(value) for value in arrays.values()}
            if len(lengths) != 1 or next(iter(lengths), 0) == 0:
                raise ValueError(f"unaligned or empty expert trace: {trace}")
            if arrays["state"].shape[1:] != (8,) or arrays["actions"].shape[1:] != (7,):
                raise ValueError(f"invalid policy tensors: {trace}")
            if any(arrays[key].shape[1:] != (128, 128, 3) for key in (
                "clean_observer_image", "policy_image", "wrist_image"
            )):
                raise ValueError(f"invalid image tensors: {trace}")
            if not np.array_equal(arrays["clean_observer_image"], arrays["policy_image"]):
                raise ValueError(f"canonical clean and policy images differ: {trace}")
            if not bool(np.all(arrays["goal_predicate_satisfied_after"][-1])):
                raise ValueError(f"accepted trace ends outside its BDDL goal: {trace}")
            if not all(
                np.isfinite(arrays[key]).all()
                for key in ("state", "actions", "sim_qpos", "sim_qvel")
            ):
                raise ValueError(f"non-finite expert trace: {trace}")
        grouped[int(plan["task_id"])].append(record)
        verified.append(str(record["id"]))

    per_task = []
    threshold = float(config["expert_acceptance"]["minimum_validation_success_fraction_per_task"])
    for task in config["tasks"]:
        task_id = int(task["task_id"])
        task_records = grouped[task_id]
        accepted = sum(bool(item.get("accepted")) for item in task_records)
        fraction = accepted / len(task_records) if task_records else 0.0
        roles = {
            role: sum(
                bool(item.get("accepted")) and item.get("role_variant") == role
                for item in task_records
            )
            for role in ("original", "counterfactual")
        }
        task_pass = fraction >= threshold and all(value > 0 for value in roles.values())
        per_task.append(
            {
                "task_id": task_id,
                "attempts": len(task_records),
                "accepted": accepted,
                "fraction": fraction,
                "accepted_by_role": roles,
                "pass": task_pass,
            }
        )
    gate_pass = all(item["pass"] for item in per_task)
    if args.phase == "validation":
        required = int(config["expert_acceptance"]["required_validation_attempts_per_task"])
        gate_pass = gate_pass and all(item["attempts"] == required for item in per_task)
    result = {
        "schema": "task-role-expert-verification-v1",
        "phase": args.phase,
        "episodes": len(records),
        "verified_accepted_episodes": len(verified),
        "tasks": per_task,
        "gate_pass": gate_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

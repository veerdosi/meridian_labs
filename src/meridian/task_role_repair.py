"""Locked planning and decision rules for the contrastive task-role repair."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_repair_config(config: Mapping[str, Any], root: Path | None = None) -> None:
    if config.get("schema") != "task-role-repair-v1":
        raise ValueError("unexpected task-role repair schema")
    tasks = list(config.get("tasks", []))
    if len(tasks) != 2:
        raise ValueError("the locked repair requires exactly two tasks")
    doses = [int(value) for value in config["selection"]["doses_new_episodes"]]
    if doses != sorted(set(doses)) or any(value <= 0 or value % (2 * len(tasks)) for value in doses):
        raise ValueError("each nested dose must contain complete pairs balanced across tasks")
    maximum_pairs_per_task = max(doses) // (2 * len(tasks))
    for task in tasks:
        variants = list(task.get("role_variants", []))
        if [variant.get("id") for variant in variants] != ["original", "counterfactual"]:
            raise ValueError(f"task {task.get('task_id')} must lock original and counterfactual roles")
        if len(task.get("contrastive_layouts", [])) != maximum_pairs_per_task:
            raise ValueError(f"task {task.get('task_id')} does not cover the maximum paired dose")
        if set(task.get("expert_profiles", {})) != {
            str(variant["commanded_object"]) for variant in variants
        }:
            raise ValueError(f"task {task.get('task_id')} lacks an expert profile for each role")
        for variant in variants:
            goal = list(variant.get("goal_predicate", []))
            if len(goal) != 3 or goal[0].lower() != "on":
                raise ValueError("each role variant must lock an On goal predicate")
            if goal[1] != variant["commanded_object"]:
                raise ValueError("the BDDL goal must name the commanded object")
            if root is not None and not (root / str(variant["bddl"])).is_file():
                raise ValueError(f"missing locked BDDL file: {variant['bddl']}")
    partitions = config["state_partitions"]
    sets = [
        set(map(int, partitions[name]))
        for name in (
            "expert_tuning_init_states",
            "expert_validation_init_states",
            "training_base_init_states",
            "untouched_target_init_states",
        )
    ]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise ValueError("expert, training, and untouched initial-state partitions overlap")


def _materialize_layout(task: Mapping[str, Any], variant: Mapping[str, Any], layout: Mapping[str, Any]) -> dict:
    if int(task["task_id"]) == 18:
        objects = {
            "chefmate_8_frypan_1": {
                "xy_m": list(layout["chefmate_8_frypan_1_xy_m"]),
                "surface": "table",
            },
            "moka_pot_1": {
                "xy_m": list(layout["moka_pot_1_xy_m"]),
                "surface": "table",
            },
        }
    elif int(task["task_id"]) == 37:
        objects = {
            str(variant["commanded_object"]): {
                "xy_m": list(layout["source_xy_m"]),
                "surface": str(layout["source_surface"]),
                "slot": "source",
            },
            str(variant["other_object"]): {
                "xy_m": list(layout["reference_xy_m"]),
                "surface": "table",
                "slot": "reference",
            },
        }
    else:
        raise ValueError(f"unsupported task-role expert task: {task['task_id']}")
    return {"layout_id": str(layout["id"]), "objects": objects}


def _plan(
    *,
    arm: str,
    task: Mapping[str, Any],
    variant: Mapping[str, Any],
    layout: Mapping[str, Any],
    init_state_index: int,
    ordinal: int,
) -> dict[str, Any]:
    materialized = _materialize_layout(task, variant, layout)
    state_hash = _canonical_hash(
        {
            "task_id": int(task["task_id"]),
            "variant": str(variant["id"]),
            "layout": materialized,
            "init_state_index": int(init_state_index),
        }
    )
    return {
        "id": f"{arm}-t{task['task_id']}-{variant['id']}-{ordinal:02d}",
        "arm": arm,
        "task_suite": str(task["suite"]),
        "task_id": int(task["task_id"]),
        "task_name": str(task["name"]),
        "role_variant": str(variant["id"]),
        "prompt": str(variant["prompt"]),
        "bddl": str(variant["bddl"]),
        "commanded_object": str(variant["commanded_object"]),
        "commanded_joint": str(variant["commanded_joint"]),
        "other_object": str(variant["other_object"]),
        "other_joint": str(variant["other_joint"]),
        "goal_predicate": list(variant["goal_predicate"]),
        "destination": str(task["destination"]),
        "init_state_index": int(init_state_index),
        "seed": int(state_hash[:8], 16) % (2**31 - 1),
        "layout": materialized,
        "state_spec_sha256": state_hash,
    }


def build_targeted_plans(config: Mapping[str, Any], dose: int) -> list[dict[str, Any]]:
    validate_repair_config(config)
    allowed = list(map(int, config["selection"]["doses_new_episodes"]))
    if dose not in allowed:
        raise ValueError(f"dose {dose} is not locked: {allowed}")
    tasks = list(config["tasks"])
    pairs_per_task = dose // (2 * len(tasks))
    maximum_pairs_per_task = max(allowed) // (2 * len(tasks))
    init_states = list(map(int, config["state_partitions"]["training_base_init_states"]))
    plans = []
    for task_offset, task in enumerate(tasks):
        variants = list(task["role_variants"])
        for pair_index, layout in enumerate(task["contrastive_layouts"][:pairs_per_task]):
            init_state = init_states[
                (task_offset * maximum_pairs_per_task + pair_index) % len(init_states)
            ]
            for variant_index, variant in enumerate(variants):
                plans.append(
                    _plan(
                        arm="targeted",
                        task=task,
                        variant=variant,
                        layout=layout,
                        init_state_index=init_state,
                        ordinal=(
                            task_offset * maximum_pairs_per_task * len(variants)
                            + pair_index * len(variants)
                            + variant_index
                        ),
                    )
                )
    if len(plans) != dose:
        raise AssertionError(f"targeted plan has {len(plans)}/{dose} episodes")
    return plans


def _uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _sample_random_layout(
    rng: np.random.Generator, task: Mapping[str, Any], identifier: str
) -> dict[str, Any]:
    bounds = task["generator_bounds"]
    minimum = float(bounds["minimum_center_separation_m"])
    for _ in range(10_000):
        if int(task["task_id"]) == 18:
            pan = [
                _uniform(rng, bounds["chefmate_8_frypan_1_x_m"]),
                _uniform(rng, bounds["chefmate_8_frypan_1_y_m"]),
            ]
            moka = [
                _uniform(rng, bounds["moka_pot_1_x_m"]),
                _uniform(rng, bounds["moka_pot_1_y_m"]),
            ]
            if math.dist(pan, moka) >= minimum:
                return {
                    "id": identifier,
                    "chefmate_8_frypan_1_xy_m": pan,
                    "moka_pot_1_xy_m": moka,
                }
        elif int(task["task_id"]) == 37:
            surfaces = bounds["source_surfaces"]
            probability = float(surfaces["table"]["probability"])
            surface = "table" if float(rng.random()) < probability else "microwave_top"
            source_bounds = surfaces[surface]
            source = [_uniform(rng, source_bounds["x_m"]), _uniform(rng, source_bounds["y_m"])]
            reference = [
                _uniform(rng, bounds["reference_x_m"]),
                _uniform(rng, bounds["reference_y_m"]),
            ]
            if math.dist(source, reference) >= minimum:
                return {
                    "id": identifier,
                    "source_surface": surface,
                    "source_xy_m": source,
                    "reference_xy_m": reference,
                }
        else:
            raise ValueError(f"unsupported task: {task['task_id']}")
    raise RuntimeError("failed to sample a valid random layout")


def build_random_plans(config: Mapping[str, Any], dose: int) -> list[dict[str, Any]]:
    validate_repair_config(config)
    allowed = list(map(int, config["selection"]["doses_new_episodes"]))
    if dose not in allowed:
        raise ValueError(f"dose {dose} is not locked: {allowed}")
    maximum_dose = max(allowed)
    tasks = list(config["tasks"])
    maximum_per_task = maximum_dose // len(tasks)
    requested_per_task = dose // len(tasks)
    init_states = list(map(int, config["state_partitions"]["training_base_init_states"]))
    rng = np.random.default_rng(int(config["state_partitions"]["random_seed"]))
    by_task: dict[int, list[dict[str, Any]]] = {}
    targeted_hashes = {
        item["state_spec_sha256"] for item in build_targeted_plans(config, maximum_dose)
    }
    for task_offset, task in enumerate(tasks):
        candidates = []
        for index in range(maximum_per_task):
            variant = task["role_variants"][index % 2]
            while True:
                layout = _sample_random_layout(rng, task, f"random-t{task['task_id']}-l{index:02d}")
                init_state = init_states[(task_offset * maximum_per_task + index) % len(init_states)]
                candidate = _plan(
                    arm="random",
                    task=task,
                    variant=variant,
                    layout=layout,
                    init_state_index=init_state,
                    ordinal=task_offset * maximum_per_task + index,
                )
                if candidate["state_spec_sha256"] not in targeted_hashes:
                    break
            candidates.append(candidate)
        by_task[int(task["task_id"])] = candidates
    plans = [
        item
        for task in tasks
        for item in by_task[int(task["task_id"])][:requested_per_task]
    ]
    if len(plans) != dose:
        raise AssertionError(f"random plan has {len(plans)}/{dose} episodes")
    # Each random episode has an independent layout ID, so it cannot accidentally form a matched pair.
    layout_ids = [item["layout"]["layout_id"] for item in plans]
    if len(layout_ids) != len(set(layout_ids)):
        raise AssertionError("random plan accidentally contains a matched counterfactual pair")
    return plans


def build_expert_development_plans(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Two role-counterbalanced smoke trajectories per task on non-training states."""
    validate_repair_config(config)
    init_states = list(map(int, config["state_partitions"]["expert_tuning_init_states"]))
    plans = []
    for task_offset, task in enumerate(config["tasks"]):
        layout = task["contrastive_layouts"][0]
        for variant_index, variant in enumerate(task["role_variants"]):
            plans.append(
                _plan(
                    arm="expert_development",
                    task=task,
                    variant=variant,
                    layout=layout,
                    init_state_index=init_states[task_offset % len(init_states)],
                    ordinal=task_offset * 2 + variant_index,
                )
            )
    return plans


def build_expert_validation_plans(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Outcome-free random layouts reserved for validating the finished controller."""
    validate_repair_config(config)
    attempts = int(config["expert_acceptance"]["required_validation_attempts_per_task"])
    init_states = list(map(int, config["state_partitions"]["expert_validation_init_states"]))
    rng = np.random.default_rng(int(config["state_partitions"]["generator_validation_seed"]))
    plans = []
    for task_offset, task in enumerate(config["tasks"]):
        for index in range(attempts):
            variant = task["role_variants"][index % 2]
            layout = _sample_random_layout(rng, task, f"validation-t{task['task_id']}-l{index:02d}")
            plans.append(
                _plan(
                    arm="expert_validation",
                    task=task,
                    variant=variant,
                    layout=layout,
                    init_state_index=init_states[(task_offset + index) % len(init_states)],
                    ordinal=task_offset * attempts + index,
                )
            )
    return plans


def select_replay_episodes(
    source_registry: Mapping[str, Any], *, dose: int, seed: int
) -> list[dict[str, Any]]:
    """Select a nested, suite-balanced replay buffer from verified official sources."""
    sources = list(source_registry.get("sources", []))
    if not sources or dose <= 0 or dose % len(sources):
        raise ValueError("replay dose must be positive and balanced across registered sources")
    per_source = dose // len(sources)
    rng = np.random.default_rng(seed)
    selected_by_source = []
    for source in sources:
        available = int(source.get("available_episodes", 0))
        if available < per_source:
            raise ValueError(f"replay source has only {available}/{per_source} episodes")
        indices = rng.permutation(available)[:per_source]
        selected_by_source.append(
            [
                {
                    "id": f"replay-{source['suite']}-t{source['task_id']}-demo{int(index)}",
                    "kind": "official_hdf5",
                    "suite": str(source["suite"]),
                    "task_id": int(source["task_id"]),
                    "prompt": str(source["prompt"]),
                    "source": str(source["source"]),
                    "source_sha256": str(source["source_sha256"]),
                    "demo": f"demo_{int(index)}",
                }
                for index in indices
            ]
        )
    # Round-robin ordering makes every smaller balanced dose a prefix of the larger selection.
    return [
        selected_by_source[source_index][episode_index]
        for episode_index in range(per_source)
        for source_index in range(len(sources))
    ]


def paired_exact_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def assess_repair_gate(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the locked paired target and regression gate to completed arm evaluations."""
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        arm = str(record["arm"])
        identifier = str(record["id"])
        if identifier in grouped[arm]:
            raise ValueError(f"duplicate {arm} result: {identifier}")
        grouped[arm][identifier] = record
    required = {"baseline", "targeted", "random"}
    if set(grouped) != required:
        raise ValueError(f"repair gate requires exactly {sorted(required)}")
    identifiers = set(grouped["baseline"])
    if any(set(grouped[arm]) != identifiers for arm in required):
        raise ValueError("arm evaluations are not paired on identical trial IDs")

    target_ids = sorted(
        identifier
        for identifier in identifiers
        if grouped["baseline"][identifier]["evaluation_suite"] == "target"
    )
    regression_ids = sorted(set(identifiers) - set(target_ids))
    expected = config["evaluation"]
    if len(target_ids) != int(expected["target_trials"]):
        raise ValueError("target result count does not match the locked plan")
    if len(regression_ids) != int(expected["regression_trials"]):
        raise ValueError("regression result count does not match the locked plan")

    success = {
        arm: sum(bool(grouped[arm][identifier]["success"]) for identifier in target_ids)
        for arm in required
    }
    task_success: dict[str, dict[int, int]] = {arm: defaultdict(int) for arm in required}
    for arm in required:
        for identifier in target_ids:
            record = grouped[arm][identifier]
            task_success[arm][int(record["task_id"])] += int(bool(record["success"]))
    wins = sum(
        bool(grouped["targeted"][identifier]["success"])
        and not bool(grouped["random"][identifier]["success"])
        for identifier in target_ids
    )
    losses = sum(
        bool(grouped["random"][identifier]["success"])
        and not bool(grouped["targeted"][identifier]["success"])
        for identifier in target_ids
    )
    p_value = paired_exact_p_value(wins, losses)
    regression_success = {
        arm: sum(bool(grouped[arm][identifier]["success"]) for identifier in regression_ids)
        for arm in required
    }
    gate = config["stopping"]["decisive_gate"]
    task_ids = sorted(task_success["baseline"])
    checks = {
        "minimum_targeted_successes": success["targeted"]
        >= int(gate["minimum_targeted_successes"]),
        "minimum_per_task": all(
            task_success["targeted"][task_id]
            >= int(gate["minimum_targeted_successes_per_task"])
            for task_id in task_ids
        ),
        "targeted_beats_random_per_task": all(
            task_success["targeted"][task_id] > task_success["random"][task_id]
            for task_id in task_ids
        ),
        "pooled_margin": success["targeted"] - success["random"]
        >= int(gate["minimum_pooled_targeted_minus_random_successes"]),
        "paired_exact": p_value <= float(gate["maximum_paired_exact_p_value"]),
        "no_regression": regression_success["baseline"] - regression_success["targeted"]
        <= int(gate["maximum_targeted_regression_losses_vs_released"]),
    }
    return {
        "schema": "task-role-repair-gate-v1",
        "successes": success,
        "wilson_95": {
            arm: wilson_interval(value, len(target_ids)) for arm, value in success.items()
        },
        "task_successes": {arm: dict(values) for arm, values in task_success.items()},
        "paired": {"targeted_wins": wins, "targeted_losses": losses, "p_value": p_value},
        "regression_successes": regression_success,
        "checks": checks,
        "decisive_pass": all(checks.values()),
    }

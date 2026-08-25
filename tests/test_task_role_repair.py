from pathlib import Path

import yaml

from meridian.task_role_repair import (
    assess_repair_gate,
    build_expert_development_plans,
    build_expert_validation_plans,
    build_random_plans,
    build_targeted_plans,
    paired_exact_p_value,
    select_replay_episodes,
    validate_repair_config,
)


def config() -> dict:
    return yaml.safe_load(Path("configs/task_role_repair_v1.yaml").read_text())


def test_locked_config_contains_real_counterfactual_pairs_and_disjoint_holdouts() -> None:
    locked = config()
    validate_repair_config(locked, Path.cwd())
    for task in locked["tasks"]:
        variants = task["role_variants"]
        assert variants[0]["commanded_object"] == variants[1]["other_object"]
        assert variants[0]["other_object"] == variants[1]["commanded_object"]
        assert variants[0]["prompt"] != variants[1]["prompt"]
        assert variants[0]["goal_predicate"][1] != variants[1]["goal_predicate"][1]


def test_targeted_plans_are_balanced_complete_pairs_and_nested() -> None:
    locked = config()
    small = build_targeted_plans(locked, 8)
    medium = build_targeted_plans(locked, 24)
    assert len(small) == 8
    assert len(medium) == 24
    assert {item["state_spec_sha256"] for item in small} <= {
        item["state_spec_sha256"] for item in medium
    }
    for task_id in (18, 37):
        task_small = [item for item in small if item["task_id"] == task_id]
        assert len(task_small) == 4
        by_layout = {}
        for item in task_small:
            by_layout.setdefault(item["layout"]["layout_id"], []).append(item)
        assert len(by_layout) == 2
        assert all(
            {item["role_variant"] for item in pair} == {"original", "counterfactual"}
            for pair in by_layout.values()
        )


def test_expert_tuning_validation_and_training_states_are_separate() -> None:
    locked = config()
    development = build_expert_development_plans(locked)
    validation = build_expert_validation_plans(locked)
    training = build_targeted_plans(locked, 24) + build_random_plans(locked, 24)
    assert len(development) == 4
    assert len(validation) == 10
    assert {item["init_state_index"] for item in development} == {25}
    assert {item["init_state_index"] for item in validation} == {26}
    assert {item["init_state_index"] for item in training} <= set(range(27, 40))


def test_random_plans_are_balanced_unpaired_reproducible_and_nested() -> None:
    locked = config()
    first = build_random_plans(locked, 8)
    second = build_random_plans(locked, 8)
    medium = build_random_plans(locked, 24)
    assert first == second
    assert {item["state_spec_sha256"] for item in first} <= {
        item["state_spec_sha256"] for item in medium
    }
    assert len({item["layout"]["layout_id"] for item in medium}) == len(medium)
    assert {item["state_spec_sha256"] for item in medium}.isdisjoint(
        {item["state_spec_sha256"] for item in build_targeted_plans(locked, 24)}
    )
    for task_id in (18, 37):
        task = [item for item in first if item["task_id"] == task_id]
        assert [item["role_variant"] for item in task].count("original") == 2
        assert [item["role_variant"] for item in task].count("counterfactual") == 2


def test_replay_is_suite_balanced_reproducible_and_nested() -> None:
    registry = {
        "sources": [
            {
                "suite": suite,
                "task_id": index,
                "prompt": f"task {index}",
                "source": f"/data/{suite}.hdf5",
                "source_sha256": str(index) * 64,
                "available_episodes": 50,
            }
            for index, suite in enumerate(
                ("libero_spatial", "libero_object", "libero_goal", "libero_10"), start=1
            )
        ]
    }
    small = select_replay_episodes(registry, dose=8, seed=440711)
    medium = select_replay_episodes(registry, dose=24, seed=440711)
    assert small == select_replay_episodes(registry, dose=8, seed=440711)
    assert {item["id"] for item in small} <= {item["id"] for item in medium}
    assert {item["suite"] for item in small} == {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    }
    assert all(sum(item["suite"] == suite for item in small) == 2 for suite in {
        item["suite"] for item in small
    })


def _records(targeted_success: set[int], random_success: set[int]) -> list[dict]:
    records = []
    for arm in ("baseline", "targeted", "random"):
        for index in range(40):
            successes = targeted_success if arm == "targeted" else random_success
            records.append(
                {
                    "arm": arm,
                    "id": f"target-{index}",
                    "evaluation_suite": "target",
                    "task_id": 18 if index < 20 else 37,
                    "success": index in successes if arm != "baseline" else False,
                }
            )
        for index in range(20):
            records.append(
                {
                    "arm": arm,
                    "id": f"regression-{index}",
                    "evaluation_suite": "regression",
                    "task_id": index % 4,
                    "success": True,
                }
            )
    return records


def test_decisive_gate_requires_paired_margin_both_tasks_and_no_regression() -> None:
    targeted = set(range(8)) | set(range(20, 28))
    random = {0, 1, 20, 21}
    assessment = assess_repair_gate(_records(targeted, random), config())
    assert assessment["paired"] == {
        "targeted_wins": 12,
        "targeted_losses": 0,
        "p_value": paired_exact_p_value(12, 0),
    }
    assert assessment["decisive_pass"] is True

    regressed = _records(targeted, random)
    next(
        item
        for item in regressed
        if item["arm"] == "targeted" and item["id"] == "regression-0"
    )["success"] = False
    assessment = assess_repair_gate(regressed, config())
    assert assessment["checks"]["no_regression"] is False
    assert assessment["decisive_pass"] is False

from pathlib import Path

from meridian.semantic_discovery import (
    build_discovery_plans,
    inventory_tasks,
    score_inventory,
    select_discovery_and_reserve,
)


def test_semantic_inventory_and_split_are_deterministic_and_disjoint(tmp_path: Path) -> None:
    task_map = tmp_path / "map.py"
    task_map.write_text(
        "libero_task_map = {'a': ['open_drawer', 'put_both_items_in_basket'], "
        "'b': ['stack_bowls', 'put_mug_left_of_plate']}\n"
    )
    for suite, names in {
        "a": ["open_drawer", "put_both_items_in_basket"],
        "b": ["stack_bowls", "put_mug_left_of_plate"],
    }.items():
        (tmp_path / suite).mkdir()
        for name in names:
            predicate = "(Open drawer_1)" if "open" in name else "(On item_1 target_1)"
            (tmp_path / suite / f"{name}.bddl").write_text(
                f"(:language {name.replace('_', ' ')})\n(:objects item_1 - item)\n"
                f"(:goal (And {predicate}))\n"
            )
    config = {
        "compatible_suites": ["a", "b"],
        "benchmark_discovery_priors": {
            "suites": {
                "a": {"failure_prior": 0.1, "reported": True},
                "b": {"failure_prior": 0.2, "reported": False},
            }
        },
        "selection": {
            "score_weights": {
                "interaction_complexity": 0.4, "semantic_rarity": 0.2,
                "benchmark_failure_prior": 0.2, "unreported_suite_uncertainty": 0.2,
            },
            "discovery_quota": {"a": 1, "b": 1},
            "confirmation_quota": {"a": 1, "b": 1},
            "discovery_count": 2,
            "confirmation_reserve_count": 2,
            "greedy_weights": {"prior_score": 0.65, "categorical_diversity": 0.35},
        },
        "discovery_screen": {
            "profiles": ["canonical", "compound_view_visual"],
            "canonical": {"replan_steps": 5.0, "action_noise": 0.0},
            "compound_view_visual": {"replan_steps": 5.0, "action_noise": 0.0},
            "seed_base": 100,
        },
    }
    inventory = inventory_tasks(task_map, tmp_path, config["compatible_suites"])
    scored = score_inventory(inventory, config)
    discovery, reserve = select_discovery_and_reserve(scored, config)
    assert len(inventory) == 4
    assert {(row["task_suite"], row["task_id"]) for row in discovery}.isdisjoint(
        {(row["task_suite"], row["task_id"]) for row in reserve}
    )
    plans = build_discovery_plans(discovery, config)
    assert len(plans) == 4
    for index in range(0, len(plans), 2):
        assert plans[index]["seed"] == plans[index + 1]["seed"]
        assert plans[index]["init_state_index"] == plans[index + 1]["init_state_index"]

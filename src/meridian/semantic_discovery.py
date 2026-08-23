from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path


def load_task_map(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("libero_task_map assignment not found")


def _section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _metadata(suite: str, task_id: int, name: str, bddl: str) -> dict:
    language_match = re.search(r"\(:language\s+(.+?)\)\s*\n", bddl)
    language = language_match.group(1).strip() if language_match else name.replace("_", " ")
    goal = _section(bddl, "goal")
    predicates = [item.lower() for item in re.findall(r"\(([A-Z][A-Za-z_]*)\s", goal)]
    predicates = [item for item in predicates if item != "and"]
    objects = re.findall(
        r"^[ \t]+([\w ]+?)[ \t]+-[ \t]+[\w]+[ \t]*$",
        _section(bddl, "objects"),
        re.MULTILINE,
    )
    object_count = sum(len(item.split()) for item in objects)
    scene_match = re.match(r"([A-Z_]+SCENE\d+)_", name)
    scene = scene_match.group(1).lower() if scene_match else suite
    lower = language.lower()
    goal_count = len(predicates)
    language_articulation = any(word in lower for word in ("open ", "close "))
    if "stack" in lower:
        capability = "stacking"
        structure = "stack_then_place" if goal_count > 1 else "stacking"
    elif any(token in predicates for token in ("turnon", "turnoff")):
        capability = "appliance_state"
        structure = "toggle_then_place" if goal_count > 1 else "toggle"
    elif (
        any(token in predicates for token in ("open", "close")) and goal_count > 1
    ) or (language_articulation and " and " in lower):
        capability = "articulated_sequence"
        structure = "articulate_then_manipulate"
    elif any(token in predicates for token in ("open", "close")):
        capability = "articulation"
        structure = "single_articulation"
    elif any(
        word in lower
        for word in ("left", "right", "front", "back", "under", "between", "next to")
    ):
        capability = "precise_spatial_relation"
        structure = "relative_placement"
    elif goal_count > 1 or "both" in lower:
        capability = "multi_object_composition"
        structure = "sequential_multi_object"
    elif re.search(r"\b(in|inside)\b", lower) or "drawer" in lower or "microwave" in lower:
        capability = "containment"
        structure = "pick_place_containment"
    elif "on " in lower or "on top" in lower:
        capability = "surface_placement"
        structure = "pick_place_surface"
    else:
        capability = "object_identity_pick_place"
        structure = "single_pick_place"
    complexity = min(
        1.0,
        0.18
        + 0.24 * max(0, goal_count - 1)
        + 0.18 * ("and" in lower or "both" in lower)
        + 0.16 * (capability in {"stacking", "articulated_sequence", "appliance_state"})
        + 0.12 * (capability == "precise_spatial_relation")
        + 0.12 * min(1.0, object_count / 8),
    )
    return {
        "task_suite": suite,
        "task_id": task_id,
        "task_name": name,
        "language": language,
        "scene": scene,
        "goal_predicates": sorted(predicates),
        "goal_count": goal_count,
        "object_count": object_count,
        "semantic_capability": capability,
        "interaction_structure": structure,
        "interaction_complexity": round(complexity, 6),
    }


def inventory_tasks(task_map_path: Path, bddl_root: Path, suites: list[str]) -> list[dict]:
    task_map = load_task_map(task_map_path)
    inventory = []
    for suite in suites:
        for task_id, name in enumerate(task_map[suite]):
            bddl_path = bddl_root / suite / f"{name}.bddl"
            if not bddl_path.is_file():
                raise FileNotFoundError(bddl_path)
            row = _metadata(suite, task_id, name, bddl_path.read_text())
            row["bddl_sha256"] = __import__("hashlib").sha256(bddl_path.read_bytes()).hexdigest()
            inventory.append(row)
    return inventory


def _features(row: dict) -> set[str]:
    return {
        f"cap:{row['semantic_capability']}",
        f"structure:{row['interaction_structure']}",
        f"scene:{row['scene']}",
        *(f"predicate:{item}" for item in row["goal_predicates"]),
    }


def _distance(left: dict, right: dict) -> float:
    a, b = _features(left), _features(right)
    return 1.0 - len(a & b) / len(a | b)


def score_inventory(inventory: list[dict], config: dict) -> list[dict]:
    counts = Counter(row["semantic_capability"] for row in inventory)
    maximum_rarity = max(1 / count for count in counts.values())
    maximum_failure = max(
        float(item["failure_prior"])
        for item in config["benchmark_discovery_priors"]["suites"].values()
    )
    weights = config["selection"]["score_weights"]
    scored = []
    for row in inventory:
        benchmark = config["benchmark_discovery_priors"]["suites"][row["task_suite"]]
        rarity = (1 / counts[row["semantic_capability"]]) / maximum_rarity
        failure = float(benchmark["failure_prior"]) / maximum_failure
        uncertainty = float(not benchmark["reported"])
        prior = (
            float(weights["interaction_complexity"]) * row["interaction_complexity"]
            + float(weights["semantic_rarity"]) * rarity
            + float(weights["benchmark_failure_prior"]) * failure
            + float(weights["unreported_suite_uncertainty"]) * uncertainty
        )
        scored.append(
            {
                **row,
                "semantic_rarity": round(rarity, 6),
                "benchmark_failure_prior": float(benchmark["failure_prior"]),
                "benchmark_result_reported": bool(benchmark["reported"]),
                "discovery_prior_score": round(prior, 6),
            }
        )
    return scored


def _greedy_select(pool: list[dict], quota: dict[str, int], weights: dict) -> list[dict]:
    chosen: list[dict] = []
    for suite, count in quota.items():
        available = [row for row in pool if row["task_suite"] == suite]
        for _ in range(int(count)):
            ranked = []
            for row in available:
                diversity = min((_distance(row, item) for item in chosen), default=1.0)
                greedy = (
                    float(weights["prior_score"]) * row["discovery_prior_score"]
                    + float(weights["categorical_diversity"]) * diversity
                )
                ranked.append((greedy, row["discovery_prior_score"], row))
            greedy_score, _, selected = min(
                ranked,
                key=lambda item: (-item[0], -item[1], item[2]["task_suite"], item[2]["task_id"]),
            )
            selected = {**selected, "selection_greedy_score": round(greedy_score, 6)}
            chosen.append(selected)
            available = [row for row in available if row["task_id"] != selected["task_id"]]
    return chosen


def select_discovery_and_reserve(scored: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    selection = config["selection"]
    available_by_suite = Counter(row["task_suite"] for row in scored)
    for suite in selection["discovery_quota"]:
        required = int(selection["discovery_quota"][suite]) + int(
            selection["confirmation_quota"][suite]
        )
        if required > available_by_suite[suite]:
            raise ValueError(
                f"discovery plus confirmation quota exceeds inventory for {suite}: "
                f"{required}>{available_by_suite[suite]}"
            )
    discovery = _greedy_select(
        scored, selection["discovery_quota"], selection["greedy_weights"]
    )
    discovery_keys = {(row["task_suite"], row["task_id"]) for row in discovery}
    remaining = [
        row for row in scored if (row["task_suite"], row["task_id"]) not in discovery_keys
    ]
    reserve = _greedy_select(
        remaining, selection["confirmation_quota"], selection["greedy_weights"]
    )
    if len(discovery) != int(selection["discovery_count"]):
        raise ValueError("discovery quota does not match locked count")
    if len(reserve) != int(selection["confirmation_reserve_count"]):
        raise ValueError("confirmation quota does not match locked count")
    return discovery, reserve


def build_discovery_plans(discovery: list[dict], config: dict) -> list[dict]:
    screen = config["discovery_screen"]
    suites = config["compatible_suites"]
    plans = []
    for task_index, task in enumerate(discovery):
        suite_index = suites.index(task["task_suite"])
        init_state = (int(task["task_id"]) * 7 + suite_index * 11) % 50
        paired_seed = int(screen["seed_base"]) + task_index
        for profile in screen["profiles"]:
            plans.append(
                {
                    "id": f"semantic-discovery-{task['task_suite']}-task{task['task_id']}-{profile}",
                    "task_suite": task["task_suite"],
                    "task_id": int(task["task_id"]),
                    "seed": paired_seed,
                    "init_state_index": float(init_state),
                    "search_stage": "semantic_discovery",
                    "stress_profile": profile,
                    **screen[profile],
                }
            )
    return plans

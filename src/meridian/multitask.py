from __future__ import annotations

from collections import defaultdict


def _profile_name(result: dict) -> str:
    return str(result.get("stress_profile") or result.get("parameters", {}).get("stress_profile"))


def build_multitask_screen_plans(config: dict) -> dict[str, list[dict]]:
    canonical = config["canonical"]
    profiles = config["profiles"]
    repeats = int(config["screen"]["repeats"])
    plans_by_suite = {}
    for suite_index, suite in enumerate(config["suites"]):
        suite_name = suite["name"]
        plans = []
        for task_id in suite["task_ids"]:
            init_state_index = float((int(task_id) * 5) % 50)
            for profile_index, (profile_name, overrides) in enumerate(profiles.items()):
                for repeat in range(repeats):
                    point = {**canonical, **overrides}
                    plans.append(
                        {
                            "id": (
                                f"multitask-screen-{suite_name}-task{task_id}-"
                                f"{profile_name}-repeat{repeat}"
                            ),
                            "task_suite": suite_name,
                            "task_id": int(task_id),
                            "seed": 12000
                            + suite_index * 1000
                            + int(task_id) * 50
                            + profile_index * 10
                            + repeat,
                            "init_state_index": init_state_index,
                            "search_stage": "screen",
                            "stress_profile": profile_name,
                            **point,
                        }
                    )
        plans_by_suite[suite_name] = plans
    return plans_by_suite


def nominate_multitask_candidates(results: list[dict], config: dict) -> list[dict]:
    """Nominate validation candidates without treating a screen failure as selection."""
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for result in results:
        grouped[(str(result["task_suite"]), int(result["task_id"]))][_profile_name(result)] = result

    priors = config["nomination"]["profile_priors"]
    candidates = []
    for (suite, task_id), profiles in sorted(grouped.items()):
        canonical = profiles.get("canonical")
        if canonical is None or not canonical["success"]:
            continue
        failed = []
        for profile_name, prior in priors.items():
            result = profiles.get(profile_name)
            if result is None or result["success"]:
                continue
            parameters = result.get("parameters", {})
            canonical_control = (
                float(parameters.get("replan_steps", 5.0))
                == float(config["canonical"]["replan_steps"])
                and float(parameters.get("action_noise", 0.0))
                == float(config["canonical"]["action_noise"])
            )
            score = (
                float(config["selection_criteria"]["intervention_headroom_weight"])
                + float(config["selection_criteria"]["canonical_control_persistence_weight"])
                * float(canonical_control)
                + float(config["selection_criteria"]["coverage_specificity_weight"])
                * float(prior["coverage_specificity"])
                + float(config["selection_criteria"]["low_random_coverage_weight"])
                * float(prior["low_random_coverage"])
            )
            failed.append((score, profile_name, canonical_control, prior))
        if not failed:
            continue
        score, profile_name, canonical_control, prior = max(failed)
        candidates.append(
            {
                "task_suite": suite,
                "task_id": task_id,
                "task": canonical.get("task"),
                "candidate_profile": profile_name,
                "screen_priority": round(score, 6),
                "screen_canonical_success": True,
                "screen_stress_failure": True,
                "canonical_control_persistence": canonical_control,
                "coverage_specificity_prior": float(prior["coverage_specificity"]),
                "low_random_coverage_prior": float(prior["low_random_coverage"]),
                "selection_status": "validation_nominee_only",
            }
        )

    candidates.sort(key=lambda row: (-row["screen_priority"], row["task_suite"], row["task_id"]))
    limit = int(config["validation"]["candidates"])
    per_suite = int(config["nomination"]["maximum_per_suite_before_global_fill"])
    selected, counts = [], defaultdict(int)
    for candidate in candidates:
        if counts[candidate["task_suite"]] < per_suite:
            selected.append(candidate)
            counts[candidate["task_suite"]] += 1
            if len(selected) == limit:
                return selected
    selected_keys = {(row["task_suite"], row["task_id"]) for row in selected}
    for candidate in candidates:
        key = (candidate["task_suite"], candidate["task_id"])
        if key not in selected_keys:
            selected.append(candidate)
            selected_keys.add(key)
            if len(selected) == limit:
                break
    return selected


def build_multitask_validation_plans(candidates: list[dict], config: dict) -> list[dict]:
    """Build paired repeated validation and family-ablation plans."""
    canonical = config["canonical"]
    profiles = config["profiles"]
    offsets = config["validation"]["initial_state_offsets"]
    plans = []
    for candidate_index, candidate in enumerate(candidates):
        profile_name = candidate["candidate_profile"]
        if profile_name == "compound_view_visual":
            conditions = ["canonical", "compound_view_visual", "viewpoint", "visual"]
        elif profile_name == "viewpoint":
            conditions = ["canonical", "viewpoint", "visual"]
        elif profile_name == "visual":
            conditions = ["canonical", "visual", "viewpoint"]
        else:
            raise ValueError(f"unsupported candidate profile: {profile_name}")
        base_init = (int(candidate["task_id"]) * 5) % 50
        for repeat, offset in enumerate(offsets):
            paired_seed = 20000 + candidate_index * 100 + repeat
            for condition in conditions:
                point = {**canonical, **profiles[condition]}
                plans.append(
                    {
                        "id": (
                            f"multitask-validation-{candidate['task_suite']}-task"
                            f"{candidate['task_id']}-{profile_name}-{condition}-repeat{repeat}"
                        ),
                        "task_suite": candidate["task_suite"],
                        "task_id": int(candidate["task_id"]),
                        "seed": paired_seed,
                        "init_state_index": float((base_init + int(offset)) % 50),
                        "search_stage": "validation",
                        "candidate_profile": profile_name,
                        "validation_condition": condition,
                        **point,
                    }
                )
    return plans


def score_validated_candidates(
    results: list[dict], candidates: list[dict], config: dict
) -> list[dict]:
    """Apply the locked eligibility, weighted score, and deterministic tie-break rule."""
    grouped: dict[tuple[str, int, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        parameters = result.get("parameters", {})
        profile = str(parameters.get("candidate_profile") or result.get("candidate_profile"))
        condition = str(
            parameters.get("validation_condition") or result.get("validation_condition")
        )
        grouped[(str(result["task_suite"]), int(result["task_id"]), profile)][condition].append(
            result
        )

    criteria = config["selection_criteria"]
    priors = config["nomination"]["profile_priors"]
    scored = []
    for candidate in candidates:
        key = (
            candidate["task_suite"],
            int(candidate["task_id"]),
            candidate["candidate_profile"],
        )
        conditions = grouped.get(key, {})
        profile = candidate["candidate_profile"]
        canonical_rows = conditions.get("canonical", [])
        stress_rows = conditions.get(profile, [])
        canonical_control_rows = [
            row
            for row in stress_rows
            if float(row.get("parameters", {}).get("replan_steps", 5.0)) == 5.0
            and float(row.get("parameters", {}).get("action_noise", 0.0)) == 0.0
        ]

        def success_rate(rows: list[dict]) -> float:
            return sum(bool(row["success"]) for row in rows) / len(rows) if rows else 0.0

        canonical_rate = success_rate(canonical_rows)
        stress_rate = success_rate(stress_rows)
        repeatability = 1.0 - stress_rate
        headroom = max(0.0, canonical_rate - stress_rate)
        control_failure_rate = 1.0 - success_rate(canonical_control_rows)
        if profile == "compound_view_visual":
            component_failure = max(
                1.0 - success_rate(conditions.get("viewpoint", [])),
                1.0 - success_rate(conditions.get("visual", [])),
            )
        else:
            other = "visual" if profile == "viewpoint" else "viewpoint"
            component_failure = 1.0 - success_rate(conditions.get(other, []))
        specificity = max(0.0, min(1.0, repeatability - component_failure))
        low_random = float(priors[profile]["low_random_coverage"])
        complete = (
            len(canonical_rows) >= int(criteria["minimum_validation_repeats"])
            and len(stress_rows) >= int(criteria["minimum_validation_repeats"])
            and len(canonical_control_rows) >= int(criteria["minimum_validation_repeats"])
        )
        eligible = (
            complete
            and canonical_rate >= float(criteria["minimum_canonical_success_rate"])
            and repeatability >= float(criteria["minimum_stress_failure_rate"])
            and headroom >= float(criteria["minimum_intervention_headroom"])
            and control_failure_rate
            >= float(criteria["minimum_canonical_control_failure_rate"])
        )
        weighted = (
            float(criteria["repeatability_weight"]) * repeatability
            + float(criteria["intervention_headroom_weight"]) * headroom
            + float(criteria["canonical_control_persistence_weight"]) * control_failure_rate
            + float(criteria["coverage_specificity_weight"]) * specificity
            + float(criteria["low_random_coverage_weight"]) * low_random
        )
        scored.append(
            {
                **candidate,
                "validation_complete": complete,
                "eligible": eligible,
                "canonical_success_rate": canonical_rate,
                "stress_failure_rate": repeatability,
                "intervention_headroom": headroom,
                "canonical_control_failure_rate": control_failure_rate,
                "coverage_specificity": specificity,
                "low_random_coverage": low_random,
                "weighted_score": weighted,
            }
        )
    return sorted(
        scored,
        key=lambda row: (
            not row["eligible"],
            -row["weighted_score"],
            -row["intervention_headroom"],
            -row["coverage_specificity"],
            -row["low_random_coverage"],
            -row["stress_failure_rate"],
            -row["canonical_success_rate"],
            row["task_suite"],
            row["task_id"],
            row["candidate_profile"],
        ),
    )

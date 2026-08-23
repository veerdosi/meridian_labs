from __future__ import annotations


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

from pathlib import Path

import yaml

from meridian.task_role_intervention import build_evaluation_plan


def test_intervention_plan_is_fixed_paired_and_complete() -> None:
    config = yaml.safe_load(Path("configs/task_role_boundary_v1/intervention.yaml").read_text())
    plans = build_evaluation_plan(config)
    target = [item for item in plans if item["evaluation_suite"] == "target"]
    regression = [item for item in plans if item["evaluation_suite"] == "regression"]
    assert len(target) == 40
    assert len(regression) == 20
    assert {item["init_state_index"] for item in target} == set(range(40, 50))
    assert all(item["max_steps"] == 400 for item in plans)
    for task_id in (18, 37):
        task_records = [item for item in target if item["task_id"] == task_id]
        assert len(task_records) == 20
        assert {item["repeat"] for item in task_records} == {0, 1}

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plan_replay_interventions import main as plan_replays
from scripts.plan_viewpoint_intervention_campaign import build_plans


def test_locked_campaign_separates_training_primary_and_confirmation() -> None:
    config = yaml.safe_load(
        (ROOT / "artifacts/manifests/campaigns/pi05-libero10-viewpoint-v1-design.yaml").read_text()
    )
    plans = build_plans(config)
    assert {name: len(rows) for name, rows in plans.items()} == {
        "source_pool": 30,
        "primary_evaluation": 80,
        "confirmation_evaluation": 40,
        "baseline_and_source": 110,
    }
    sources = plans["source_pool"]
    primary = plans["primary_evaluation"]
    confirmation = plans["confirmation_evaluation"]
    assert {row["init_state_index"] for row in sources} == set(range(30))
    assert {row["init_state_index"] for row in confirmation} == set(range(40, 50))
    assert {row["seed"] for row in sources}.isdisjoint(row["seed"] for row in primary)
    assert {row["seed"] for row in sources}.isdisjoint(row["seed"] for row in confirmation)
    target = [row for row in primary if row["evaluation_suite"] == "target"]
    assert len(target) == 40
    assert all(0.08 <= row["camera_x"] <= 0.12 for row in target)
    assert all(32 <= row["camera_yaw_deg"] <= 38 for row in target)
    assert all(row["occlusion"] == 0 and row["visual_distractors"] == 0 for row in target)


def test_replay_plans_match_sources_and_only_intervene_on_viewpoint(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = yaml.safe_load(
        (ROOT / "artifacts/manifests/campaigns/pi05-libero10-viewpoint-v1-design.yaml").read_text()
    )
    sources = build_plans(campaign)["source_pool"]
    source_path = tmp_path / "sources.jsonl"
    source_path.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "success": True,
                    "parameters": {"evaluation_suite": "training_source"},
                }
            )
            + "\n"
            for row in sources
        )
    )
    output = tmp_path / "plans"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_replay_interventions.py",
            "--config",
            str(ROOT / "configs/pi05_libero_viewpoint_intervention_v1.yaml"),
            "--interventions",
            str(ROOT / "artifacts/manifests/interventions/pi05-libero10-viewpoint-d8-v1.yaml"),
            "--source-rollouts",
            str(source_path),
            "--output",
            str(output),
        ],
    )
    plan_replays()
    arms = {
        arm: [json.loads(line) for line in (output / f"{arm}.jsonl").read_text().splitlines()]
        for arm in ("targeted", "random", "original_distribution")
    }
    assert all(len(rows) == 8 for rows in arms.values())
    assert [row["source_rollout_id"] for row in arms["targeted"]] == [
        row["source_rollout_id"] for row in arms["random"]
    ] == [row["source_rollout_id"] for row in arms["original_distribution"]]
    for rows in arms.values():
        assert all(
            row["brightness"] == 1
            and row["occlusion"] == 0
            and row["visual_distractors"] == 0
            for row in rows
        )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["targeted"]["target_component_plans"] == 6
    assert all(summary[arm]["sources_matched_across_arms"] for arm in arms)

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_policy_rollouts_write_all_required_evidence_streams() -> None:
    source = (ROOT / "scripts/openpi_libero_rollout.py").read_text()
    for name in ("clean_observer.mp4", "policy_input.mp4", "wrist.mp4", "diagnostic.mp4"):
        assert name in source
    for key in ("clean_observer_image", "policy_image", "wrist_image"):
        assert key in source
    assert '"image": videos.get("diagnostic")' not in source
    assert '"video": videos.get("diagnostic")' in source


def test_replay_uses_same_recording_contract() -> None:
    source = (ROOT / "scripts/replay_visual_intervention.py").read_text()
    assert "observer_pose = camera_pose(env)" in source
    assert "clean_image = render_at_camera_pose(env, observer_pose)" in source
    assert "write_evidence_videos(" in source
    assert "clean_observer_image=clean_images" in source
    assert "policy_image=policy_images" in source

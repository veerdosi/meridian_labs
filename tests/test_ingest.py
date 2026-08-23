import json
from pathlib import Path

from meridian.ingest import ingest_libero_rollouts
from meridian.models import Budget, ExperimentSpec, ParameterAxis, ParameterSpace


def test_ingest_libero_rollout(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        policy_adapter="openpi",
        checkpoint_id="pi05_libero",
        checkpoint_source="gs://checkpoint",
        task_suite="libero_spatial",
        regression_suites=["canonical"],
        evaluation_seeds=[1],
        budget=Budget(max_rollouts=1),
        parameter_space=ParameterSpace(
            axes=[ParameterAxis(name="camera_x", low=-1, high=1, semantic="view")]
        ),
    )
    raw = {
        "id": "episode-1",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "task": "pick up",
        "seed": 1,
        "init_state_index": 0,
        "parameters": {"camera_x": 0.2},
        "success": True,
        "score": 1.0,
        "phase_labels": ["complete"],
        "steps": 10,
        "duration_seconds": 2.0,
        "inference_seconds": 1.0,
        "trace": "/scratch/trace.npz",
        "trace_sha256": "abc",
        "video": "/scratch/video.mp4",
    }
    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps(raw) + "\n")
    records = ingest_libero_rollouts(spec, source)
    assert records[0].parameters == {"camera_x": 0.2}
    assert records[0].artifacts[0].sha256 == "abc"

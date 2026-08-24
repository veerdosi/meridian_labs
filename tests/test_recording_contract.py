from pathlib import Path

import numpy as np
import pytest

from meridian.recording import diagnostic_frames, write_evidence_videos


def frames(count: int = 2) -> list[np.ndarray]:
    return [np.full((16, 16, 3), index * 40, dtype=np.uint8) for index in range(count)]


def test_diagnostic_frames_are_synchronized_and_labeled() -> None:
    result = diagnostic_frames(
        frames(), frames(), frames(), parameters={"init_state_index": 7}, success=False
    )
    assert result.shape == (2, 64, 48, 3)
    assert result.dtype == np.uint8


def test_diagnostic_frames_reject_misaligned_streams() -> None:
    with pytest.raises(ValueError, match="identical lengths"):
        diagnostic_frames(frames(2), frames(1), frames(2), parameters={}, success=True)


def test_evidence_writer_creates_all_four_videos(tmp_path: Path) -> None:
    paths = write_evidence_videos(
        tmp_path, frames(), frames(), frames(), parameters={"replan_steps": 5}, success=True
    )
    assert set(paths) == {"clean_observer", "policy_input", "wrist", "diagnostic"}
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths.values())

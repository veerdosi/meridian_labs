from pathlib import Path

import pytest

from meridian.adapters.openpi import OpenPILiberoAdapter


def test_server_command_pins_checkpoint() -> None:
    adapter = OpenPILiberoAdapter(port=8123)
    adapter.load("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero")
    command = adapter.server_command(Path("/env/python"), Path("/src/openpi"))
    assert "--policy.config=pi05_libero" in command
    assert "--policy.dir=gs://openpi-assets/checkpoints/pi05_libero" in command
    assert "--port=8123" in command


def test_unbounded_training_is_rejected() -> None:
    adapter = OpenPILiberoAdapter()
    with pytest.raises(RuntimeError, match="bounded scheduler trainer"):
        adapter.finetune(None, "intervention", 10, 1, Path("/tmp"))  # type: ignore[arg-type]

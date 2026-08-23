from pathlib import Path

from meridian.openpi_overlay import BEGIN, MARKER, install


def test_overlay_is_idempotent(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text("import logging\n\n_CONFIGS = [\n" + MARKER + "]\n")
    install(config)
    once = config.read_text()
    install(config)
    assert config.read_text() == once
    assert once.count(BEGIN) == 1
    assert "import os\n" in once
    assert "pi05_libero/params" in once

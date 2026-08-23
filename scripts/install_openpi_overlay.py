#!/usr/bin/env python3
"""Install Meridian's environment-driven fine-tuning config into a pinned OpenPI checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.openpi_overlay import install


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("openpi_root", type=Path)
    args = parser.parse_args()
    target = args.openpi_root / "src/openpi/training/config.py"
    install(target)
    print(f"installed Meridian training config in {target}")


if __name__ == "__main__":
    main()

"""Deterministic selection for the original-distribution training control."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

ORIGINAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_TASKS_PER_SUITE = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def _rank(seed: int, namespace: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{identifier}".encode()).hexdigest()


def _task_prompt(path: str) -> str:
    stem = PurePosixPath(path).stem.removesuffix("_demo")
    if "_SCENE" in stem:
        stem = stem.split("_SCENE", 1)[1].split("_", 1)[1]
    return stem.replace("_", " ")


def select_original_distribution_episodes(
    catalog: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    maximum_dose: int,
    dataset_root: str,
    demos_per_task: int = 50,
) -> list[dict[str, Any]]:
    """Select a nested, suite-balanced sample from π0.5's four LIBERO training suites."""
    if maximum_dose < len(ORIGINAL_SUITES) or maximum_dose % len(ORIGINAL_SUITES):
        raise ValueError("maximum dose must be a positive multiple of four")
    if demos_per_task < 1:
        raise ValueError("demos_per_task must be positive")
    files = []
    for item in catalog:
        path = str(item.get("path", ""))
        if item.get("type") != "file" or not path.endswith(".hdf5"):
            continue
        suite = path.split("/", 1)[0]
        lfs = item.get("lfs") or {}
        sha256 = str(lfs.get("oid", ""))
        if len(sha256) != 64:
            raise ValueError(f"missing LFS SHA-256 for {path}")
        files.append(
            {
                "path": path,
                "suite": suite,
                "size_bytes": int(item["size"]),
                "source_sha256": sha256,
            }
        )
    counts = Counter(item["suite"] for item in files)
    if counts != Counter(EXPECTED_TASKS_PER_SUITE):
        raise ValueError(f"unexpected LIBERO task catalog: {dict(sorted(counts.items()))}")

    per_suite = maximum_dose // len(ORIGINAL_SUITES)
    selected_by_suite = {}
    for suite in ORIGINAL_SUITES:
        selected_by_suite[suite] = sorted(
            (item for item in files if item["suite"] == suite),
            key=lambda item: (
                _rank(seed, "original-task", item["path"]),
                item["path"],
            ),
        )[:per_suite]

    selected = [
        selected_by_suite[suite][round_index]
        for round_index in range(per_suite)
        for suite in ORIGINAL_SUITES
    ]
    return [
        {
            "demo": f"demo_{int(_rank(seed, 'episode', item['path']), 16) % demos_per_task}",
            "source": f"{dataset_root.rstrip('/')}/{item['path']}",
            "source_path": item["path"],
            "source_sha256": item["source_sha256"],
            "source_size_bytes": item["size_bytes"],
            "suite": item["suite"],
            "task": _task_prompt(item["path"]),
        }
        for item in selected
    ]

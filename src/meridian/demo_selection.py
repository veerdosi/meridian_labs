"""Deterministic demonstration selection with explicit state-partition protection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def select_partitioned_maximin(
    demo_features: Mapping[str, Sequence[float]],
    inventory_features: Mapping[int, Sequence[float]],
    *,
    allowed_indices: Sequence[int],
    forbidden_indices: Sequence[int],
    count: int,
) -> list[dict[str, Any]]:
    """Map demos to natural states, exclude protected partitions, then space-fill."""
    if count < 1:
        raise ValueError("count must be positive")
    inventory_indices = sorted(int(index) for index in inventory_features)
    inventory = np.asarray([inventory_features[index] for index in inventory_indices], dtype=float)
    if inventory.ndim != 2 or inventory.shape[1] == 0:
        raise ValueError("inventory features must be a non-empty two-dimensional table")
    allowed = {int(index) for index in allowed_indices}
    forbidden = {int(index) for index in forbidden_indices}
    if allowed & forbidden:
        raise ValueError("allowed and forbidden state partitions overlap")
    ranges = np.ptp(inventory, axis=0)
    variable = ranges > 1e-9
    if not variable.any():
        raise ValueError("inventory features do not vary")
    normalized_inventory = inventory[:, variable] / ranges[variable]

    nearest_by_state: dict[int, dict[str, Any]] = {}
    for identifier in sorted(demo_features):
        feature = np.asarray(demo_features[identifier], dtype=float)
        if feature.shape != (inventory.shape[1],):
            raise ValueError(f"demo {identifier} has an incompatible feature shape")
        normalized = feature[variable] / ranges[variable]
        distances = np.linalg.norm(normalized_inventory - normalized, axis=1)
        nearest_offset = int(np.argmin(distances))
        nearest_index = inventory_indices[nearest_offset]
        if nearest_index in forbidden or nearest_index not in allowed:
            continue
        candidate = {
            "demo": identifier,
            "nearest_init_state_index": nearest_index,
            "nearest_normalized_distance": float(distances[nearest_offset]),
            "normalized_feature": normalized,
        }
        previous = nearest_by_state.get(nearest_index)
        if previous is None or (
            (candidate["nearest_normalized_distance"], identifier)
            < (previous["nearest_normalized_distance"], previous["demo"])
        ):
            nearest_by_state[nearest_index] = candidate
    candidates = list(nearest_by_state.values())
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} partition-safe unique demos; {count} required")

    selected = [
        min(candidates, key=lambda item: (item["nearest_normalized_distance"], item["demo"]))
    ]
    remaining = [item for item in candidates if item is not selected[0]]
    while len(selected) < count:
        chosen = min(
            remaining,
            key=lambda item: (
                -min(
                    float(
                        np.linalg.norm(
                            item["normalized_feature"] - existing["normalized_feature"]
                        )
                    )
                    for existing in selected
                ),
                item["demo"],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return [
        {key: value for key, value in item.items() if key != "normalized_feature"}
        for item in selected
    ]

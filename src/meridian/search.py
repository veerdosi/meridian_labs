from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from meridian.adapters.base import PolicyAdapter
from meridian.models import CapabilityMap, ExperimentSpec, RegionSummary, RolloutRecord
from meridian.store import ExperimentStore


def _wilson(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    if count == 0:
        return (0.0, 1.0)
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * count)) / count) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def propose_parameter_points(
    spec: ExperimentSpec, prior: list[RolloutRecord], count: int, seed: int
) -> list[dict[str, float]]:
    """Propose a scheduler-friendly batch, adapting toward nearby opposing outcomes."""
    rng = np.random.default_rng(seed)

    def random_point() -> dict[str, float]:
        return {
            axis.name: float(rng.uniform(axis.low, axis.high)) for axis in spec.parameter_space.axes
        }

    def normalized(record: RolloutRecord) -> np.ndarray:
        return np.asarray(
            [
                (record.parameters[axis.name] - axis.low) / (axis.high - axis.low)
                for axis in spec.parameter_space.axes
            ]
        )

    points = []
    working = list(prior)
    successes = [record for record in working if record.success]
    failures = [record for record in working if not record.success]
    for _ in range(count):
        if not successes or not failures:
            points.append(random_point())
            continue
        pairs = (
            (float(np.linalg.norm(normalized(success) - normalized(failure))), success, failure)
            for success in successes
            for failure in failures
        )
        _, success, failure = min(pairs, key=lambda item: item[0])
        point = {}
        for index, axis in enumerate(spec.parameter_space.axes):
            midpoint = (success.parameters[axis.name] + failure.parameters[axis.name]) / 2
            jitter = rng.normal(0, 0.035) * (axis.high - axis.low)
            point[axis.name] = float(np.clip(midpoint + jitter, axis.low, axis.high))
        points.append(point)
    return points


class AdaptiveFailureSearch:
    def __init__(self, spec: ExperimentSpec, adapter: PolicyAdapter, store: ExperimentStore):
        self.spec = spec
        self.adapter = adapter
        self.store = store

    def _random_point(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            axis.name: float(rng.uniform(axis.low, axis.high))
            for axis in self.spec.parameter_space.axes
        }

    def _normalized(self, rollout: RolloutRecord) -> np.ndarray:
        return np.asarray(
            [
                (rollout.parameters[a.name] - a.low) / (a.high - a.low)
                for a in self.spec.parameter_space.axes
            ]
        )

    def _boundary_point(
        self, rollouts: list[RolloutRecord], rng: np.random.Generator
    ) -> dict[str, float]:
        successes = [r for r in rollouts if r.success]
        failures = [r for r in rollouts if not r.success]
        if not successes or not failures:
            return self._random_point(rng)
        candidates: list[tuple[float, RolloutRecord, RolloutRecord]] = []
        for success in successes:
            s = self._normalized(success)
            for failure in failures:
                candidates.append(
                    (float(np.linalg.norm(s - self._normalized(failure))), success, failure)
                )
        _, success, failure = min(candidates, key=lambda item: item[0])
        jitter = rng.normal(0, 0.025, len(self.spec.parameter_space.axes))
        point = {}
        for index, axis in enumerate(self.spec.parameter_space.axes):
            midpoint = (success.parameters[axis.name] + failure.parameters[axis.name]) / 2
            point[axis.name] = float(
                np.clip(midpoint + jitter[index] * (axis.high - axis.low), axis.low, axis.high)
            )
        return point

    def run(
        self, count: int | None = None, seed: int = 0, warmup: int | None = None
    ) -> list[RolloutRecord]:
        count = min(count or self.spec.budget.max_rollouts, self.spec.budget.max_rollouts)
        warmup = warmup or max(8, 2 * len(self.spec.parameter_space.axes))
        rng = np.random.default_rng(seed)
        rollouts: list[RolloutRecord] = []
        for index in range(count):
            parameters = (
                self._random_point(rng) if index < warmup else self._boundary_point(rollouts, rng)
            )
            rollout = self.adapter.rollout(self.spec, parameters, seed * 1_000_000 + index)
            self.store.put("rollout", rollout)
            rollouts.append(rollout)
        self.store.event(
            self.spec.id, "failure_search.completed", {"count": len(rollouts), "seed": seed}
        )
        return rollouts


def build_capability_map(
    spec: ExperimentSpec, rollouts: list[RolloutRecord], bins: int = 3
) -> CapabilityMap:
    grouped: dict[tuple[int, ...], list[RolloutRecord]] = defaultdict(list)
    for rollout in rollouts:
        key = []
        for axis in spec.parameter_space.axes:
            value = (rollout.parameters[axis.name] - axis.low) / (axis.high - axis.low)
            key.append(min(bins - 1, max(0, int(value * bins))))
        grouped[tuple(key)].append(rollout)
    regions = []
    for key, members in sorted(grouped.items()):
        successes = sum(member.success for member in members)
        low, high = _wilson(successes, len(members))
        bounds = {}
        for index, axis in enumerate(spec.parameter_space.axes):
            width = (axis.high - axis.low) / bins
            bounds[axis.name] = (axis.low + key[index] * width, axis.low + (key[index] + 1) * width)
        regions.append(
            RegionSummary(
                bounds=bounds,
                count=len(members),
                successes=successes,
                success_rate=successes / len(members),
                wilson_low=low,
                wilson_high=high,
                evidence_rollout_ids=[member.id for member in members],
            )
        )
    boundary_pairs = []
    successes = [r for r in rollouts if r.success]
    failures = [r for r in rollouts if not r.success]
    for success in successes:
        if not failures:
            break

        def distance(failure: RolloutRecord, successful: RolloutRecord = success) -> float:
            return sum(
                ((successful.parameters[a.name] - failure.parameters[a.name]) / (a.high - a.low))
                ** 2
                for a in spec.parameter_space.axes
            )

        nearest = min(failures, key=distance)
        if distance(nearest) < 0.08:
            boundary_pairs.append((success.id, nearest.id))
    capability_map = CapabilityMap(
        experiment_id=spec.id,
        rollout_count=len(rollouts),
        global_success_rate=sum(r.success for r in rollouts) / len(rollouts),
        regions=regions,
        failure_clusters=sorted(
            (r for r in regions if r.success_rate < 0.5), key=lambda r: (r.success_rate, -r.count)
        ),
        boundary_pairs=boundary_pairs,
    )
    return capability_map

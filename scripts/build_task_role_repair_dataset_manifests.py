#!/usr/bin/env python3
"""Build equal-dose targeted/random manifests with an identical fixed replay buffer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from meridian.task_role_repair import select_replay_episodes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expert_episodes(plans_path: Path, results_path: Path, arm: str, dose: int) -> list[dict]:
    plans = load_jsonl(plans_path)
    records = load_jsonl(results_path)
    if len(plans) != dose or len(records) != dose:
        raise ValueError(f"{arm} generation has {len(plans)}/{len(records)}/{dose} plan/result/dose")
    if [item["id"] for item in plans] != [item["id"] for item in records]:
        raise ValueError(f"{arm} results do not match the locked plan order")
    episodes = []
    for plan, record in zip(plans, records):
        if plan["arm"] != arm or record["arm"] != arm or not bool(record.get("accepted")):
            raise ValueError(f"{arm} contains a rejected or mislabeled expert episode: {plan['id']}")
        if plan["state_spec_sha256"] != record["state_spec_sha256"]:
            raise ValueError(f"{arm} state specification changed during generation: {plan['id']}")
        episodes.append(
            {
                "id": str(plan["id"]),
                "kind": "expert_npz",
                "arm": arm,
                "task_id": int(plan["task_id"]),
                "role_variant": str(plan["role_variant"]),
                "prompt": str(plan["prompt"]),
                "trace": str(record["trace"]),
                "trace_sha256": str(record["trace_sha256"]),
                "state_spec_sha256": str(plan["state_spec_sha256"]),
            }
        )
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dose", type=int, required=True)
    parser.add_argument("--targeted-plans", type=Path, required=True)
    parser.add_argument("--targeted-results", type=Path, required=True)
    parser.add_argument("--random-plans", type=Path, required=True)
    parser.add_argument("--random-results", type=Path, required=True)
    parser.add_argument("--replay-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse manifest output: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text())
    if args.dose not in map(int, config["selection"]["doses_new_episodes"]):
        raise ValueError("dose is not locked")
    registry = json.loads(args.replay_registry.read_text())
    replay = select_replay_episodes(
        registry, dose=args.dose, seed=int(config["replay"]["seed"])
    )
    arms = {
        "targeted": expert_episodes(
            args.targeted_plans, args.targeted_results, "targeted", args.dose
        ),
        "random": expert_episodes(args.random_plans, args.random_results, "random", args.dose),
    }
    args.output_dir.mkdir(parents=True)
    summary = []
    for arm, new_episodes in arms.items():
        manifest = {
            "schema": "task-role-repair-dataset-manifest-v1",
            "arm": arm,
            "dose_new": args.dose,
            "dose_replay": len(replay),
            "config": str(args.config),
            "config_sha256": file_sha256(args.config),
            "replay_registry": str(args.replay_registry),
            "replay_registry_sha256": file_sha256(args.replay_registry),
            "episodes": new_episodes + replay,
        }
        path = args.output_dir / f"{arm}-dose{args.dose}-with-replay.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        summary.append(
            {
                "arm": arm,
                "path": str(path),
                "sha256": file_sha256(path),
                "new": len(new_episodes),
                "replay": len(replay),
            }
        )
    targeted_replay = json.loads(Path(summary[0]["path"]).read_text())["episodes"][args.dose :]
    random_replay = json.loads(Path(summary[1]["path"]).read_text())["episodes"][args.dose :]
    if targeted_replay != random_replay:
        raise AssertionError("targeted and random replay buffers differ")
    print(json.dumps({"schema": "task-role-repair-manifests-v1", "outputs": summary}, sort_keys=True))


if __name__ == "__main__":
    main()

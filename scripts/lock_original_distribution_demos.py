#!/usr/bin/env python3
"""Lock nested original-distribution LIBERO control episodes."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from meridian.control_selection import select_original_distribution_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--doses", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    doses = sorted(set(args.doses))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {args.output_dir}")
    url = (
        f"https://huggingface.co/api/datasets/{args.repository}/tree/"
        f"{args.revision}?recursive=true&limit=1000"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        catalog = json.load(response)
    selected = select_original_distribution_episodes(
        catalog,
        seed=args.seed,
        maximum_dose=max(doses),
        dataset_root=args.dataset_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "meridian-task-role-original-distribution-selection-v1",
        "repository": args.repository,
        "revision": args.revision,
        "seed": args.seed,
        "doses": doses,
        "selection": (
            "seeded task rank, suite-balanced across Spatial, Object, Goal, and LIBERO-10"
        ),
        "maximum_dose_episodes": selected,
    }
    (args.output_dir / "original-distribution-selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    for dose in doses:
        manifest = {
            "schema": "meridian-libero-hdf5-episode-manifest-v1",
            "arm": "original_distribution",
            "dose": dose,
            "episodes": selected[:dose],
        }
        (args.output_dir / f"original_distribution-dose-{dose}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
    print(json.dumps({"doses": doses, "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()

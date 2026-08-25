#!/usr/bin/env python3
"""Download and hash only the four fixed replay sources used by the repair campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import yaml
from huggingface_hub import hf_hub_download


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    if args.registry.exists():
        raise FileExistsError(f"refusing to overwrite source registry: {args.registry}")
    spec = yaml.safe_load(args.spec.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = []
    for source in spec["sources"]:
        downloaded = Path(
            hf_hub_download(
                repo_id=str(spec["repository"]),
                repo_type="dataset",
                revision=str(spec["revision"]),
                filename=str(source["filename"]),
                local_dir=args.output_root,
            )
        ).resolve()
        with h5py.File(downloaded, "r") as archive:
            demos = sorted(name for name in archive["data"] if name.startswith("demo_"))
            if not demos:
                raise ValueError(f"source contains no demonstrations: {downloaded}")
            for demo in demos:
                group = archive[f"data/{demo}"]
                if not bool(group["rewards"][-1]) or not bool(group["dones"][-1]):
                    raise ValueError(f"source contains an unsuccessful episode: {downloaded}:{demo}")
        sources.append(
            {
                **source,
                "source": str(downloaded),
                "source_sha256": file_sha256(downloaded),
                "size_bytes": downloaded.stat().st_size,
                "available_episodes": len(demos),
            }
        )
    registry = {
        "schema": "task-role-replay-source-registry-v1",
        "repository": spec["repository"],
        "revision": spec["revision"],
        "spec": str(args.spec),
        "spec_sha256": file_sha256(args.spec),
        "sources": sources,
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps(registry, sort_keys=True))


if __name__ == "__main__":
    main()

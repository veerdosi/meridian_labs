#!/usr/bin/env python3
"""Apply the preregistered primary or confirmation gate for task-3 viewpoint data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meridian.viewpoint_gate import evaluate_confirmation_gate, evaluate_primary_gate


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--stage", choices=("dose8", "dose24"))
    parser.add_argument("--confirmation-targeted", type=Path)
    parser.add_argument("--confirmation-random", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.confirmation_targeted or args.confirmation_random:
        if not all((args.confirmation_targeted, args.confirmation_random)):
            raise SystemExit("confirmation mode requires both confirmation arms")
        payload = evaluate_confirmation_gate(
            primary_targeted=load_jsonl(args.targeted),
            primary_random=load_jsonl(args.random),
            confirmation_targeted=load_jsonl(args.confirmation_targeted),
            confirmation_random=load_jsonl(args.confirmation_random),
        )
    else:
        if args.baseline is None or args.stage is None:
            raise SystemExit("primary mode requires --baseline and --stage")
        payload = evaluate_primary_gate(
            baseline=load_jsonl(args.baseline),
            targeted=load_jsonl(args.targeted),
            random=load_jsonl(args.random),
            original=load_jsonl(args.original) if args.original else None,
            stage=args.stage,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": payload["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()

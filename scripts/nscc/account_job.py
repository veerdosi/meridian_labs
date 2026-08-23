#!/usr/bin/env python3
"""Convert a finished PBS qstat record into the Meridian cost schema."""

from __future__ import annotations

import argparse
import json
import re
import subprocess


def parse_duration(value: str) -> float:
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    hours, minutes, seconds = value.split(":")
    return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    args = parser.parse_args()
    text = subprocess.check_output(["qstat", "-xf", args.job_id], text=True)
    fields: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"\s*([\w.]+)\s*=\s*(.*)", line)
        if match:
            current = match.group(1)
            fields[current] = match.group(2).strip()
        elif current and line.startswith("\t"):
            fields[current] += line.strip()
    requested_wall = parse_duration(fields.get("Resource_List.walltime", "00:00:00"))
    actual_wall = parse_duration(fields.get("resources_used.walltime", "00:00:00"))
    ngpus = int(fields.get("Resource_List.ngpus", "0"))
    ncpus = int(fields.get("Resource_List.ncpus", "0"))
    gpu_hours = ngpus * actual_wall / 3600
    cpu_hours = 0.0 if ngpus else ncpus * actual_wall / 3600
    payload = {
        "job_id": args.job_id,
        "queue": fields.get("queue"),
        "project": fields.get("project"),
        "requested_ncpus": ncpus,
        "requested_ngpus": ngpus,
        "requested_walltime_seconds": requested_wall,
        "actual_walltime_seconds": actual_wall,
        "cpu_core_hours": cpu_hours,
        "gpu_hours": gpu_hours,
        "su": gpu_hours * 64 if ngpus else cpu_hours,
        "exit_status": int(fields.get("Exit_status", "-1")),
        "source": "pbs_qstat_finished_record",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

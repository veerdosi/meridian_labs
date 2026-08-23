from __future__ import annotations

from meridian.models import ResourceCost


def resource_cost_from_manifests(manifests: list[dict]) -> ResourceCost:
    if not manifests:
        raise ValueError("at least one job manifest is required")
    requested = [manifest["requested"] for manifest in manifests]
    actual = [manifest["actual"] for manifest in manifests]
    queues = {str(manifest["queue"]) for manifest in manifests}
    return ResourceCost(
        job_id="+".join(str(manifest["job_id"]) for manifest in manifests),
        queue=next(iter(queues)) if len(queues) == 1 else "+".join(sorted(queues)),
        requested_ncpus=sum(int(item["ncpus"]) for item in requested),
        requested_ngpus=sum(int(item["ngpus"]) for item in requested),
        requested_walltime_seconds=sum(int(item["walltime_seconds"]) for item in requested),
        actual_walltime_seconds=sum(float(item["walltime_seconds"]) for item in actual),
        gpu_hours=sum(float(item["walltime_seconds"]) for item in actual) / 3600,
        su=sum(float(item["su_estimated"]) for item in actual),
        source=(
            "nscc_pbs_finished_record"
            if len(manifests) == 1
            else "nscc_pbs_finished_records"
        ),
    )

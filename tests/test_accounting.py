from meridian.accounting import resource_cost_from_manifests


def manifest(job_id: str, walltime: int, su: float) -> dict:
    return {
        "job_id": job_id,
        "queue": "gdev",
        "requested": {"ncpus": 16, "ngpus": 1, "walltime_seconds": 1800},
        "actual": {"walltime_seconds": walltime, "su_estimated": su},
    }


def test_combines_repaired_evaluation_accounting() -> None:
    cost = resource_cost_from_manifests(
        [manifest("partial", 900, 16.0), manifest("repair", 450, 8.0)]
    )
    assert cost.job_id == "partial+repair"
    assert cost.requested_ngpus == 2
    assert cost.actual_walltime_seconds == 1350
    assert cost.gpu_hours == 0.375
    assert cost.su == 24.0
    assert cost.source == "nscc_pbs_finished_records"

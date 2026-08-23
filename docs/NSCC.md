# NSCC execution

The harness uses ASPIRE 2A through PBS Pro. All simulation, inference, and training work runs as
scheduler jobs; the login node is used only for source transfer, environment inspection, job
submission, and result retrieval.

## Cost model

ASPIRE 2A charges GPU jobs at 64 SU per requested GPU card-hour and does not additionally charge
their fixed CPU allocation. CPU-only jobs cost 1 SU per requested CPU core-hour. PBS pre-charges
requested walltime and refunds unused time after completion. `scripts/nscc/account_job.py` reads the
finished `qstat -xf` record and emits the versioned cost shape used by the experiment store.

The project's 500 SU starting guidance equals 7.8125 one-GPU hours. It is an initial profiling
target, not a project cap. Before each wider search or training campaign, estimate requested SU from
the measured jobs it expands.

## Storage

- Source/runtime: `/scratch/users/ntu/veer001/meridian_labs`
- OpenPI cache: `/scratch/users/ntu/veer001/meridian_labs/cache/openpi`
- Scratch quota: 100 TB, purgeable
- Home quota: 50 GB; model caches must not be placed there

Every useful cluster result is copied back to the external SSD with a small versioned manifest.


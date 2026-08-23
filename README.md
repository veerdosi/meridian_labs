# Meridian Labs

Meridian is a policy-agnostic capability cartographer and intervention harness. The reference
deployment targets Physical Intelligence's released `pi05_libero` checkpoint in parameterized
LIBERO/MuJoCo on NSCC ASPIRE 2A.

The authoritative scope and definition of done are in [`BUILD_BRIEF.md`](BUILD_BRIEF.md). A
repository setup, baseline, or smoke test is a gate—not the research result.

## Local contract test

```bash
uv sync --extra dev
uv run pytest
uv run meridian surrogate-e2e --output artifacts/surrogate-e2e
```

The surrogate run exercises adaptive failure search, capability mapping, competing hypotheses,
targeted/random/original/oracle intervention arms, uncertainty, regression checks, provenance, and
report generation. It validates plumbing only and must never be reported as evidence about π0.5.

## Real π0.5 result

The first controlled NSCC campaign is recorded in
[`artifacts/results/pi05-spatial-intervention-v1/RESULT.md`](artifacts/results/pi05-spatial-intervention-v1/RESULT.md).
The harness found a repeatable compound visual/viewpoint boundary and identified short replanning as
a causal failure amplifier. All four matched 100-step LoRA variants improved over the released
checkpoint without regression, but the selected narrow targeted arm (80%) did not beat random (95%)
or original-distribution data (90%). The honest decision is therefore **do not scale the narrow
targeted intervention**; refine the targeting model and test whether the broad-data gain generalizes.

The result bundle includes typed evaluations and training runs, paired comparisons, cost accounting,
the machine-readable decision, and a SQLite experiment store. Rebuild it deterministically from the
versioned manifests and SSD-backed rollout bundles with:

```bash
uv run python scripts/record_real_campaign.py \
  --runs artifacts/manifests/campaigns/pi05-spatial-intervention-v1-runs.yaml \
  --output artifacts/results/pi05-spatial-intervention-v1
```

The campaign used 80.14 SU; cumulative measured build and research usage was 152.90 SU.

The sequential dose-8 follow-up is recorded in
[`artifacts/results/pi05-spatial-intervention-v2/RESULT.md`](artifacts/results/pi05-spatial-intervention-v2/RESULT.md).
An evidence-weighted 6-target/2-broad mixture reached 40/40 target success and 20/20 regression,
versus 37/40 for matched random data, 8/40 for original-distribution data, and 21/40 for the
released checkpoint. The locked targeted-versus-random gate was positive, although its paired
sign test remains non-decisive (3 wins, 0 losses, p=0.25); targeted versus original was decisive
(32 wins, 0 losses). Campaign v2 used 102.49 SU and brought cumulative measured usage to 255.39 SU.

## NSCC

Cluster jobs and accounting helpers are under `scripts/nscc/`; infrastructure and cost conventions
are documented in [`docs/NSCC.md`](docs/NSCC.md). All model caches and large artifacts use NSCC
scratch or the external SSD. Credentials never belong in this repository.

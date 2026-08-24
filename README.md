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

## Active real-policy direction

The final experiment uses LIBERO's predefined physical initial states, synchronized MuJoCo
telemetry, and canonical π0.5 control. It must identify a repeatable stage-specific boundary,
infer the missing data region, and test targeted data against matched random and original-data
controls at multiple doses. Screening and training are not launched until the locked protocol and
implementation pass a final review.

Superseded campaign details are intentionally absent from the active tree. The concise history and
recovery reference are in [`artifacts/LEGACY_CAMPAIGNS.md`](artifacts/LEGACY_CAMPAIGNS.md).

## NSCC

Cluster jobs and accounting helpers are under `scripts/nscc/`; infrastructure and cost conventions
are documented in [`docs/NSCC.md`](docs/NSCC.md). All model caches and large artifacts use NSCC
scratch or the external SSD. Credentials never belong in this repository.

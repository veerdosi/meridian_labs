# Legacy campaign archive

Detailed pre-final manifests, plans, raw results, audits, and PBS records were removed from
the active tree to keep the final harness maintainable. They remain recoverable from Git tag
`legacy-campaign-archive-2026-08-24` (commit `00520ac`).

Essential context:

- Task 1 completed an end-to-end visual intervention pilot, but retrospective review found
  that its camera/mask condition removed critical observer information. Treat it as qualified
  engineering evidence, not the final scientific demonstration.
- The earlier multi-task semantic discovery/validation records are retained as negative engineering
  evidence: semantic and synthetic visual failures were not sufficiently physical, specific, or
  causally grounded for the final claim.
- Detailed Task 2 v1 (`pi05-libero10-viewpoint-*`) artifacts were deliberately removed from live
  SSD/NSCC storage and from this archive tag. The campaign is not part of the research evidence.
- Measured cumulative NSCC usage through those campaigns was 436.9041655521 SU.
- The active final direction uses naturally valid LIBERO initial states, physical simulator
  telemetry, stage-level diagnosis, targeted/random/original data controls, dose stopping,
  and untouched confirmation states.

Inspect an archived artifact without restoring it using
`git show legacy-campaign-archive-2026-08-24:<path>`.

# Physical-boundary campaign: final launch review

## Claim under test

For a valid, naturally occurring LIBERO initial-state region where released `pi05_libero` is
otherwise competent, stage-linked targeted simulation data improves untouched boundary states more
than the same amount of random or original-distribution data, without more than 2 percentage points
of regression.

This experiment tests **physical initial/contact-state coverage**. It does not test every possible
data gap and does not claim universal root-cause diagnosis.

## Engineering preflight

PBS job `15246052` ran one noncampaign LIBERO Goal trajectory from init state 17 for 25 policy
steps. It exited successfully and passed the physical rollout verifier:

- exact policy input, clean observer view, wrist view, and diagnostic video: present and synchronized;
- action, robot/object state, contact, and exact BDDL predicate telemetry: finite and aligned;
- interpretable physical initial-state features: 25;
- object, receptacle, robot, and task workspace: visible, with no synthetic camera transform or
  information-destroying mask;
- actual wall time: 1,256 seconds; actual charge: 22.3289 SU.

The short rollout did not complete the task. That is not a scientific outcome: its 25-step horizon
was chosen only to validate the measuring instrument. The telemetry reported 6.52 cm end-effector
motion, no robot-scene contact, negligible object motion, and 0/1 final goal predicates.

Local evidence is stored under ignored SSD path `artifacts/physical-preflight/15246052/`.

## Locked experiment

| Stage | Exact work | Gate |
| --- | --- | --- |
| Inventory | Read physical features for the predefined state partitions; no policy rollouts | Every required state resolves and no goal is already satisfied |
| Screen | 60 canonical rollouts: 10 tasks × 6 natural initial states, balanced across Spatial, Object, Goal, LIBERO-10, and LIBERO-90 | Candidate needs ≥3/6 successes, ≥1/6 failures, ≥0.60 physical specificity, and estimated random coverage ≤0.35 |
| Confirm | Top two candidates: 3 exact repeats, replanning controls at 1 and 10 steps, and 7 disjoint confirmation states each | All repeats and controls retain the failure; predicted-success minus predicted-failure success is ≥0.50; four untouched failure-side states exist |
| Diagnose | Compare BDDL progress, end-effector/object motion, contact, articulation, and videos | State the observed failure stage and competing data/non-data explanations; telemetry alone cannot prove cause |
| Collect | 15 predefined training-source states for the selected task | At least eight unique successful sources; targeted selection uses the locked physical feature/threshold |
| Dose 4 | Train targeted and random, then release equal-dose original only if targeted beats random; evaluate every arm on the same 40 target + 20 regression trials | Stop if targeted does not strictly beat either control or regression loss exceeds 0.02 |
| Dose 8 | Run only after a positive dose-4 gate; use nested sources and the identical evaluation plans | Select dose 8 only if it adds ≥2/40 target successes; otherwise stop at dose 4 |
| Oracle | Not automatic | Run only if a concrete unresolved scientific question remains |

Every comparison reports paired outcomes, Wilson 95% intervals, PBS accounting, and regressions.
Untouched states 40–49 are not inspected before the boundary rule is fixed.

## Cost and stopping

Measured cumulative usage before the campaign is 459.2331 SU, including this preflight. The old
60-SU screening figure is guidance, not a ceiling. The 25-step preflight spent roughly 12 minutes on
policy startup and 2 minutes on rollout, so a 60-rollout full-horizon screen is likely to cost
materially more than 60 SU. A conservative timeout-heavy estimate is approximately 250–350 SU;
early task completion would reduce it. Submit screening as its own stage, record actual accounting,
and do not reserve training SU until a boundary passes confirmation and no-data controls.

## What would be paper-worthy

A positive result requires targeted data to beat both equal-dose controls on the untouched boundary,
remain within the regression limit, and reproduce at the selected dose. A negative result remains an
honest harness result but is not evidence that targeted physical-state data works; it triggers a new
boundary search without further training on this boundary.

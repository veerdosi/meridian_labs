# π0.5 / LIBERO intervention result

## Learned boundary and mechanism

The deployed `pi05_libero` policy had 40/40 successes in the initial envelope, then 2/24 failures in the wider search. One compound view/visual profile reproduced in 4/5 trials. Matched probes identified two-step replanning as the causal amplifier: 0/5 success with short replanning versus 3/5 with canonical replanning.

Preregistered hypothesis: Uniform data sampled from the selected compound visual/viewpoint boundary improves held-out short-replanning performance more than equal-size random and original-distribution data.

## Controlled comparison

| Arm | Target success | Wilson 95% CI | Regression | SU |
|---|---:|---:|---:|---:|
| random | 19/20 (95%) | [76%, 99%] | 100% (+0%) | 14.04 |
| oracle_targeted | 19/20 (95%) | [76%, 99%] | 100% (+0%) | 15.56 |
| original_distribution | 18/20 (90%) | [70%, 97%] | 100% (+0%) | 15.82 |
| targeted | 16/20 (80%) | [58%, 92%] | 100% (+0%) | 16.30 |
| none | 9/20 (45%) | [26%, 66%] | 100% (+0%) | 8.85 |

## Decision

Targeted data changed target success versus no intervention by +35%, with lift versus matched random of -15% and versus original-distribution data of -10%.

The preregistered targeted-selection hypothesis is falsified at this dose.

On paired plans, targeted versus random had 1 wins and 4 losses (two-sided sign-test p=0.375); the point difference is not itself proof that random is superior.

Decision engine: Do not collect more data; targeted data has not beaten all fair baselines.

Campaign compute cost was 80.14 SU; cumulative build-and-research usage was 152.90 SU.

## Next action

Test whether broad-data gains persist on more tasks and seeds, and refine the cartographer's intervention-value model so it can prefer broad coverage when the evidence does not justify narrow targeting. Retain canonical replanning as a no-training mitigation benchmark.

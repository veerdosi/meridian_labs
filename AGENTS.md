# Working rules

## Experimental engineering discipline

- Derive the terminal condition from the real benchmark, evaluator, or external contract before implementing a workflow. Stop as soon as that condition is satisfied.
- Prefer the smallest mechanism that can test the scientific claim. Do not add post-success behavior, realism, compatibility layers, audit machinery, preflights, or fallback phases unless a concrete requirement or observed failure makes them necessary.
- When an implementation fails, diagnose the first violated invariant. Fix that mechanism directly instead of expanding the surrounding system.
- Keep controls matched only on variables needed for causal interpretation. Extra variation is not automatically stronger evidence.
- Separate required evidence from nice-to-have evidence. Complete the required path first, then add optional artifacts only when they materially improve interpretation.
- Before introducing a new stage, state what decision it changes. If it changes no decision and satisfies no contract, omit it.

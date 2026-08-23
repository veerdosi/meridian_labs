# Robotics Policy Intervention Harness — Build Brief

## Purpose

Build a harness that discovers what a frontier robot policy is bad at, identifies the most likely cause, and experimentally determines the highest-value fix under a fixed experimental budget.

The first version is for vision-language-action (VLA) policies in simulation. It is not a generic robotics platform, a data-labeling tool, or a hardware/sensor diagnostic system.

The north star is a policy-agnostic **capability cartographer and intervention scientist**: a system that continually maps the boundary of a robot policy's competence, reduces uncertainty about the cause of failure, and spends experimental budget on the next action with the greatest expected capability value. A deployed pi0.5 policy in a parameterized MuJoCo/LIBERO environment is the first environment in which to establish that system, not the limit of its ambition.

## Core question

Given a policy checkpoint, a target capability, and a budget, answer:

> Where does the policy fail, why does it fail, and which next intervention is most likely to improve the target capability without unacceptable regression?

Valid answers include a targeted data-collection order, a training/configuration change, or **do not collect more data**. Hardware and sensor changes are out of scope for the initial version.

## V1 decision scope

The full product ultimately makes five linked decisions. V1 intentionally implements them at different depths:

| Decision | V1 depth | V1 interpretation |
| --- | --- | --- |
| Where is the gap? | **Core** | Search the simulator to locate and cluster failure boundaries across task, object, environment, viewpoint, initial state, temporal sequence, and recovery conditions. |
| What data is missing? | **Core** | Choose the highest-value missing coverage region and diversity requirements for simulator-generated trajectories. |
| How much is required? | **Limited but real** | Use sequential dose tests (zero/small/medium additions) to estimate marginal value and decide whether more data is justified. Do not promise a universal saturation law in v1. |
| Which source should provide it? | **Fixed** | Simulation is the only intervention source in v1. Existing datasets, teleoperation, human video, autonomous practice, cross-embodiment data, and new sensors come later. |
| Is data even the answer? | **Narrow but required** | Compare targeted data with no-data fixes available in this stack: observation/action adapter correction, normalization, camera/task configuration, inference/control settings, and training configuration. Hardware and sensor redesign are excluded. |

Thus V1 is not merely a two-decision classifier. Its deep focus is gap location plus data-content choice; it also performs a lightweight quantity decision and a bounded data-versus-non-data decision. It holds the data-source choice constant to simulation.

## First demonstration

Use the released **`pi05_libero` π0.5 checkpoint in LIBERO/MuJoCo** as the first policy/environment pair. pi0.5 is the policy checkpoint, MuJoCo is the physics/rendering engine, and LIBERO supplies a compatible robot/task/observation/action environment. This gives the harness a functioning, competent deployment baseline rather than an uncalibrated pre-training checkpoint. It is the reference implementation for proving the harness and measuring the compute cost of rollouts, inference, fine-tuning, and evaluation.

The initial research subject is the deployed `pi05_libero` policy itself. The harness must search beyond fixed benchmark scores to discover its genuine boundary under parameterized task, visual, viewpoint, state, temporal, and recovery variations. LIBERO is a starting environment and task specification; it does not require the intervention data to be copied from a fixed benchmark dataset. Generate or collect the selected intervention data in simulation and record its provenance.

Do not plant a hidden training-data gap or predeclare a toy failure as the main result. Start from the given deployed policy, search the simulator for actual failures, and let evidence select the first capability target and intervention. Compare the selected intervention against matched random and broad-distribution data additions, with the same training budget and starting checkpoint. The raw `pi05_base` checkpoint remains useful later for controlled training experiments, but should not be the first deployed subject.

The demonstration must compare the harness's selected intervention with:

- no intervention;
- the same-size random data addition;
- more data from the original distribution;
- an oracle targeted intervention when practical.

The harness should outperform random collection on the hidden out-of-distribution failure region while reporting performance on the original regression suite.

## North star and initial proof

The immediate objective is one honest, reproducible result showing whether an agent can find a non-trivial failure region of pi0.5 and select a more valuable intervention than naive random data collection. This is the first proof of the larger system, not a reason to stop once it succeeds.

A small standard LIBERO/MuJoCo task is allowed only as plumbing and cost-accounting smoke test. It is **not** the research demonstration. The real test case must be selected after a structured failure search finds a meaningful, repeatable boundary with headroom for improvement.

After the first proof, continue to deepen the system: richer failure dimensions, better causal hypothesis tests, adaptive intervention selection, marginal-return modeling, and broader task families. Preserve the same experiment contract so each new result enlarges the shared capability map rather than becoming a one-off benchmark script.

**Continuation mandate:** do not mark the goal complete after repository setup, NSCC setup, a baseline run, or the smoke test. Treat each as a gate that unlocks the next build stage. Continue autonomously toward a functioning end-to-end harness unless blocked by a genuine missing credential, unavailable resource, cost decision, or user-required choice. Keep the user informed through concise progress updates and cost reporting.

## Definition of done for the initial build

The initial build is complete only when the repository contains a working, reproducible pi0.5 + LIBERO/MuJoCo harness that can:

1. run and store parameterized evaluation rollouts;
2. construct a capability/failure map from those rollouts;
3. present evidence to the Codex scientific agent and record its competing hypotheses;
4. turn the selected hypothesis into a targeted simulator-data intervention;
5. train controlled policy variants and evaluate target gain, regression, cost, and uncertainty against fair baselines; and
6. produce a concise research result/artifact that states what was learned and what should run next.

The smoke test is **Gate 0**: it validates compute, policy, simulator, data, and accounting plumbing before building the full loop above.

## Execution environment and handoff

### Local workspace and Git

- Working directory: `/Volumes/VEER/meridian_labs`
- Remote repository: `https://github.com/veerdosi/meridian_labs` (default branch: `main`)
- The external SSD `VEER` is removable. At the beginning of work, verify that it is mounted. If it is absent, stop rather than silently writing large data to the MacBook.
- Versioned source, experiment manifests, configuration, documentation, metric summaries, and selected small artifacts belong in Git and should be pushed regularly. This makes critical work recoverable even when the SSD is disconnected.
- Large, regenerable artifacts belong on the SSD and must be Git-ignored: raw datasets, downloaded models, simulator assets, checkpoints, rollout videos, tensorboard/W&B exports, and temporary caches. Store a manifest/checksum/metadata record for every such artifact in Git.
- Do not put passwords, API keys, SSH private keys, or NSCC credentials in the repository, experiment logs, shell history, or issue text.

MuJoCo is installed on the MacBook. Use it locally for fast environment inspection, task development, unit tests, visual debugging, and lightweight rollouts when practical. Do not assume local Mac execution is representative of the NSCC GPU runtime; use NSCC for pi0.5 GPU inference/training and scalable, recorded experimental batches.

### Checkpoint and artifact placement

- Do **not** download large model checkpoints into the MacBook's default cache (for example, `~/.cache/openpi`); internal free space is limited.
- For NSCC GPU work, download the pi0.5 checkpoint directly to the project-accessible NSCC scratch/cache path discovered during the environment audit. Configure OpenPI's `OPENPI_DATA_HOME` to that path so model downloads, caches, and checkpoints remain on cluster storage while jobs run.
- For any optional local model copy or downloaded rollout artifact, use `/Volumes/VEER/meridian_labs/artifacts/model-cache/` or a sibling ignored artifact directory on the external SSD. Never commit model weights or large run artifacts to Git.
- Record checkpoint source, revision/identifier, checksum when available, and cluster/local path in a versioned experiment manifest; the path itself is not a substitute for provenance.

The local `meridian_labs` directory existed but was not yet a Git working tree when this brief was updated. Initialize/clone it only if still needed, preserving any user files already present.

### NSCC ASPIRE 2A

- Compute target: NSCC ASPIRE 2A NVIDIA A100 nodes, accessed through `veer001@aspire2antu.nscc.sg`.
- Jobs must go through the scheduler; never run simulation/training on the login node.
- Initial SU guidance: aim to use approximately **500 SU** for environment audit, resource profiling, and the first smoke test. This is not a project-level cap and does not limit in-scope follow-on work. Track burn rate carefully; before a materially more expensive sweep or training campaign, report the observed cost and expected additional use so the user can judge the trade-off.
- Record the requested resources, actual wall time, reported GPU/CPU card hours, and SU usage for every job.
- Active access is via a temporary password-authenticated SSH control connection on the user's Mac. Test it before any NSCC operation:

  ```bash
  ssh -o BatchMode=yes \
    -o ControlMaster=auto \
    -o ControlPath="$HOME/.ssh/controlmasters/nscc-%C" \
    veer001@aspire2antu.nscc.sg 'hostname'
  ```

  If the command returns a hostname, proceed. If it fails, the eight-hour connection has probably expired or been closed. Ask the user to open a fresh control connection; do not request, store, echo, or attempt to recover their password.
- Keep active simulator/training data on NSCC storage while a job is running. Copy only useful summaries and selected artifacts back to the external SSD. Treat NSCC scratch as purgeable, not as the sole copy of a result.

### Model access

The v1 scientific loop runs interactively as a Codex skill/workflow. No GPT-5.6 Sol API key is needed to build or validate it. GPT-5.6 Sol API access is a later option only if we decide to run an unattended orchestration service.

## Operating loop

1. **Define the experiment.** Register policy/checkpoint, task, budget, target success metric, regression limits, and fixed holdout suites.
2. **Map failures.** Run parameterized simulator rollouts that vary object pose/appearance, camera/viewpoint, occlusion, distractors, and robot initial state. Save traces, videos, parameters, phase labels, and success/failure outcomes.
3. **Form hypotheses.** GPT-5.6 Sol reviews the evidence and proposes competing explanations, including data-coverage gaps and training/configuration issues. Each hypothesis must state a predicted discriminating result.
4. **Run cheap tests.** Execute low-cost probes or small, targeted data injections to reject weak hypotheses.
5. **Select an intervention.** The agent produces a machine-readable intervention spec: source, data budget, targeted parameter region, diversity requirements, quality checks, and expected gain. It may recommend no new data.
6. **Generate and validate data.** Collect simulator trajectories matching the spec, verify their coverage/quality, convert them to the policy's training format, and record provenance.
7. **Train and evaluate.** Train the selected intervention and fair comparison variants with matched training budgets and starting conditions, then evaluate target success, fixed regression suites, run cost, and uncertainty across seeds where affordable.
8. **Decide and remember.** Estimate marginal value versus the baselines, record regressions and confidence, update the capability map, and recommend the next action.

## Required components

### 1. Policy adapter

A thin, swappable adapter with methods to:

- load and serve a versioned checkpoint;
- receive simulator observations and return actions;
- train or fine-tune on a dataset;
- evaluate a checkpoint on a named suite;
- expose run logs, checkpoint IDs, and resource cost.

Do not embed pi0.5- or LIBERO-specific assumptions above this adapter layer.

### 2. Failure-search engine

Defines a task parameter schema and generates/schedules rollouts over it. It must support adaptive search toward the policy's success/failure boundary, not only a static benchmark sweep.

Outputs a **capability map**: performance by parameter region, failure clusters, evidence links, and confidence.

### 3. Experiment store

Persist every rollout, dataset, training run, model version, configuration, metric, cost, and decision. Record requested CPU/GPU resources, wall time, and NSCC SU consumption where available. Reproducibility is a product requirement: every reported gain must link to its data and comparison runs.

### 4. Agentic data scientist

GPT-5.6 Sol is the scientific planner, not the policy controller. It reads summaries and artifacts, writes hypotheses and experiment specs, selects the next action under budget, and explains its evidence. Execution must be tool-mediated and bounded by explicit budget/approval rules.

### 5. Intervention/evaluation harness

Creates targeted simulation data, runs training variants, and evaluates them against target and regression suites. It must make fair comparisons possible: matched training budgets, fixed evaluation seeds, and clear baselines.

## What counts as success

For a single hidden failure region, the system can:

1. locate and describe the boundary;
2. identify a useful data-coverage hypothesis;
3. spend a fixed budget on a targeted simulator intervention;
4. show a credible target gain over random/original-distribution data; and
5. report regressions, cost, and uncertainty honestly.

## Explicit non-goals for v1

- physical robot collection or deployment;
- hardware, sensor, or embodiment redesign;
- claiming universal root-cause diagnosis;
- fully autonomous, unbounded experiment execution;
- support for every policy family before one end-to-end result works.

## Gate 0: first implementation milestone (do not stop here)

Complete a smoke test before building the agent:

1. Verify the Git workspace and active NSCC control connection, then inspect scheduler, storage, module/CUDA, and project-allocation details without consuming compute.
2. Run the released `pi05_libero` checkpoint through its supported policy adapter on one minimal LIBERO/MuJoCo task, aiming to keep the audit/smoke-test phase near 500 SU.
3. Save trajectories, videos, evaluation results, and rollout/inference resource use.
4. Generate a tiny simulator trajectory dataset and validate the data-conversion/fine-tuning path needed for future interventions.
5. Re-evaluate a minimal adapted policy checkpoint while recording training resource use separately. This verifies the intervention mechanism; it must not be presented as the main scientific result. If the smoke-test estimate exceeds the initial guidance, report the estimate and move to a deliberately costed next phase rather than treating 500 SU as a permanent stop condition.

This smoke test establishes the end-to-end contract and an initial NSCC cost profile before the agent begins selecting interventions.

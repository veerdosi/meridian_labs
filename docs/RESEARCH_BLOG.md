# We Put Codex Around a Robot Policy and Asked It What Data to Collect

π0.5 already knew how to control a robot. We gave Codex a harder job: watch the policy work, find a
repeatable failure, work out what experience might help, choose a small training set, and test that
choice against other ways of spending the same data budget.

The setup had three parts. π0.5 produced actions from a language instruction, two camera views, and
robot state. LIBERO and MuJoCo executed those actions with a simulated Panda arm. Codex planned the
experiments, inspected videos and physical traces, tested explanations for each failure, selected
data, and managed the evaluation and training runs.

## The questions we started with

Our brief contained three questions.

1. **Where is the gap?** A policy can fail during perception, localization, approach, contact,
   control, recovery, sequencing, memory, strategy selection, or termination.
2. **What kind of data is missing?** The missing coverage might involve tasks, objects,
   environments, viewpoints, initial states, contact states, strategies, recovery, time, or
   embodiment.
3. **How much data is enough?** We wanted a marginal improvement curve and a stopping rule, not an
   open-ended fine-tuning budget.

A success bit says whether the task finished. It does not say why the robot failed or what data
would help. We therefore gave Codex synchronized evidence from inside each rollout, fixed state
partitions, and control experiments that could challenge its diagnosis.

## Our first result did not survive its own video

The first intervention pilot completed the whole software path. We created a visual condition,
collected targeted trajectories, trained controlled variants, and measured a numerical gain. Then
we inspected representative videos.

A black mask covered roughly a quarter of the observer image. A camera transform pushed the robot,
the manipulated bowl, and the goal plate outside the observer field of view. The policy also used a
wrist-camera stream that our evidence video had not recorded. We could not establish that the
target condition preserved enough information to perform the task, and we could not inspect every
image the policy had used.

We dropped the result. The training code had run correctly, but the experiment had removed
information the robot needed.

Codex changed the protocol after the audit. A candidate condition now has to keep the robot,
manipulated object, receptacle, and goal region visible enough to act. Every rollout records the
exact policy input alongside the clean observer view and wrist view. Before a failure can become a
training target, it must repeat, survive simple control probes, and point to a specific kind of
missing experience.

That false start changed the rest of the project. From then on, the video had to support the claim
before we spent compute training a fix.

## Rebuilding the rollout record

We made every rollout save four synchronized visual streams:

1. the clean observer view;
2. the exact observer image sent to π0.5;
3. the wrist-camera image sent to π0.5; and
4. a diagnostic video with the views, rollout parameters, and final outcome together.

We also recorded actions, robot state, MuJoCo joint positions, object poses, robot-scene contacts,
and the exact LIBERO BDDL goal predicates at every step. The harness hashes initial simulator
states, plans, and trajectory records. It versions seeds and state partitions before Codex sees the
outcomes.

![A three-panel preflight montage showing the clean observer view, exact policy input, and wrist-camera view.](../artifacts/physical-preflight/15246052/diagnostic-montage.png)

_Figure 1. The recording preflight. This campaign uses canonical images, so the clean observer view
and policy input match. The wrist view is π0.5's second visual input. We checked synchronization,
finite actions and state, contacts, object motion, and task predicates before starting the screen._

Video shows whether the right object remains visible and what the motion looks like. Joint and
contact traces measure which body moved, when contact began, and whether a completed relation later
broke. Codex uses both when it diagnoses a rollout.

The first diagnostic rules covered concrete patterns:

- end-effector motion without contact;
- contact without meaningful object motion;
- target motion followed by loss of an achieved relation;
- target-object motion versus distractor-object motion; and
- physical progress without satisfaction of the BDDL predicate.

These measurements let Codex narrow a failure to a stage of the task, then use a control run to
separate plausible causes.

## Screening 60 natural initial states

After the recording preflight, we ran π0.5 on 60 natural initial states. The screen contained ten
tasks with six states per task. We selected two tasks from each of LIBERO Spatial, Object, Goal,
LIBERO-10, and LIBERO-90.

All 60 rollouts used canonical observations and canonical π0.5 control. We applied no masks, camera
transforms, or image post-processing. Before inference, a state-only inventory checked 500 initial
states, 50 per task, and confirmed that none began with its goal already satisfied.

We separated the states before looking at results:

| Partition         | Role                            | State indices                    |
| ----------------- | ------------------------------- | -------------------------------- |
| Discovery         | Find a candidate behavior       | 0, 3, 6, 9, 12, 15               |
| Confirmation      | Probe the selected behavior     | 18 and 21 for each selected task |
| Training source   | Supply candidate demonstrations | 25 to 39                         |
| Untouched holdout | Measure final target return     | 40 to 49                         |

π0.5 completed 48 of the 60 discovery rollouts. The task-level split was sharper than the overall
80 percent success rate:

| Suite          | Task                                          | Successes |
| -------------- | --------------------------------------------- | --------: |
| LIBERO Spatial | black bowl between plate and ramekin to plate |       6/6 |
| LIBERO Spatial | black bowl from top drawer to plate           |       6/6 |
| LIBERO Object  | alphabet soup to basket                       |       6/6 |
| LIBERO Object  | milk to basket                                |       6/6 |
| LIBERO Goal    | open the middle drawer                        |       6/6 |
| LIBERO Goal    | cream cheese to bowl                          |       6/6 |
| LIBERO-10      | turn on stove, then place moka pot            |       6/6 |
| LIBERO-10      | book to back compartment of caddy             |       6/6 |
| LIBERO-90      | frying pan to stove                           |       0/6 |
| LIBERO-90      | white bowl to right of plate                  |       0/6 |

Our first scoring rule looked for a physical boundary within a task, such as object poses that
separated successful and failed states. It expected a mix of successes and failures. These two
tasks scored poorly because each was 0/6, even though they contained every failure in the screen.

Codex stopped the pose search and opened the traces. The robot used the full 400-step horizon in
every failed rollout. Its end effector travelled about 1.1 to 1.8 metres, and it spent substantial
time in contact with objects. The failures contained purposeful motion, so we followed that motion.

## The first diagnosis measured the wrong object

Our diagnostic code initially summarized the most-moved free object in the scene. That is a
reasonable generic motion feature, but it is wrong for a relational task when the policy
manipulates a distractor. The summary can call the rollout "progress" precisely because the wrong
object moved.

The videos and BDDL predicates exposed the mismatch. Codex changed the extractor to read the target
object from the goal predicate, match it to the corresponding MuJoCo joint, and report target and
distractor displacement separately. We added integration coverage using actual rollout records and
trajectory archives, not only synthetic fixtures.

After the correction, the same pattern appeared in both tasks: the commanded object stayed nearly
still while another object moved.

## What the robot actually did

For **"put the frying pan on the stove,"** the BDDL predicate names the frying pan as the target.
The frying pan's translation was effectively zero in all six discovery states. In five trials the
moka pot moved beyond the locked 2.5-centimetre diagnostic threshold, travelling about 19 to 30
centimetres. It moved 2.43 centimetres in the remaining trial, which we keep as ambiguous rather
than rounding upward.

For **"put the white bowl to the right of the plate,"** the predicate assigns the white bowl as the
target and the plate as the reference. In five trials the bowl moved less than one millimetre while
the plate moved about 3.4 to 22 centimetres. Both objects moved in the sixth trial, so that trace is
also ambiguous.

![Representative frames from the two failed tasks, showing the requested target and the object the policy actually manipulated.](../artifacts/physical-campaign/screening/15246322/representative-videos/failure-montage.png)

_Figure 2. Canonical failures from the two selected tasks. π0.5 interacts with the moka pot instead
of the frying pan, and moves the plate instead of treating it as the relational reference._

Representative synchronized rollouts:

- [Frying pan to stove, discovery state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i0-diagnostic.mp4)
- [Frying pan to stove, discovery state 3](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i3-diagnostic.mp4)
- [White bowl to right of plate, discovery state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i0-diagnostic.mp4)
- [White bowl to right of plate, discovery state 9](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i9-diagnostic.mp4)

Ten of the 12 failures meet the strict signature: negligible target motion and substantial motion
of the distractor or reference. We marked the other two as ambiguous. Across the two tasks, π0.5
approaches, contacts, and moves an object, but chooses the wrong one.

Several causes could produce that behavior. The policy may bind the nouns in the instruction to
the wrong objects, or reuse a familiar strategy that starts with the moka pot or plate. It may also
correct the early choice if given more time or a different replanning interval. We reserved unseen
states to test those possibilities before training.

## Choosing the corrective data

If the behavior persists under those controls, we will train on successful executions of these
exact instructions. Those examples show which object plays each role while preserving the
approach, grasp, transport, and placement behavior π0.5 already uses.

### Where the demonstrations come from

LIBERO already provides 50 human-teleoperated MuJoCo demonstrations for each of the two tasks. We
downloaded the two task files from the official dataset and verified every source file by hash. We
also checked that the recorded episodes end with the real LIBERO success predicate satisfied.

Human operators created the motor commands in these demonstrations. Codex chooses which successful
episodes answer the diagnosed failure. The selector draws a balanced set from the two task files,
keeps it separate from confirmation and holdout states, and spreads the selected episodes across
the available starting geometries. The conversion step preserves the action, robot-state,
observer-camera, and wrist-camera streams in π0.5's training format. During the preflight we found
that the raw HDF5 camera arrays need a double flip to match the canonical images seen by π0.5, so
the converter now verifies orientation as well as shapes and values.

Humans supplied successful control trajectories in the simulator. Codex found the behavioral
pattern, chose the relevant subset, and will test whether that subset buys more improvement than
an equal number of randomly chosen or original-distribution demonstrations.

### A future experiment without human demonstrations

Some future environments will not come with successful demonstrations. In that case, the harness
would first need a simulator expert that can produce and verify the corrective trajectories. For
tasks like these, Codex could construct a task-specific scripted expert using MuJoCo's privileged
state:

1. Parse the LIBERO goal predicate to identify the target object and destination.
2. Read the exact target and receptacle poses from MuJoCo.
3. Generate a collision-safe approach pose above the correct object.
4. Use inverse kinematics or LIBERO's operational-space controller to approach, grasp, lift,
   transport, and place.
5. Verify success using the real BDDL goal predicate.
6. Reject unsuccessful or unstable trajectories.
7. Record observer images, wrist images, robot state, actions, contacts, and provenance in π0.5's
   training format.
8. Add controlled variation across training-only object poses and robot initial states.

The success predicate provides the acceptance test for each generated trajectory. Only successful,
stable executions would enter the training set. Building and validating this expert is a separate
experiment, but the telemetry and data-conversion path already in the harness are the pieces it
would rely on.

## The matched experiment

Every trained arm starts from the same released π0.5 checkpoint. We keep the demonstration count,
optimizer schedule, training budget, and evaluation states fixed. Only the source of the added data
changes.

| Arm                   | Added data                                                           | Measurement                                     |
| --------------------- | -------------------------------------------------------------------- | ----------------------------------------------- |
| Released checkpoint   | None                                                                 | Return from π0.5 without added training         |
| Targeted              | Balanced successful examples from the two diagnosed tasks            | Value of data selected from the trace diagnosis |
| Random                | An equal number of successful examples sampled without the diagnosis | Value of generic extra demonstrations           |
| Original distribution | An equal number from π0.5's familiar LIBERO distribution             | Value of more in-distribution fine-tuning       |

The random arm uses the same number of trajectories, sampled without the object-selection
diagnosis. The original-distribution arm uses the same number from tasks already familiar to the
released checkpoint. We fix all source records before training.

We evaluate dose 4 first. Dose 8 runs only if dose 4 passes the locked target-gain and regression
gate. Final target evaluation uses 40 untouched trials, balanced across the two tasks. A separate
20-trial plan measures regression on established capabilities. Shared initial states support
paired comparisons, and Wilson intervals show uncertainty around each binomial success rate.

The final table will compare target return, regression, added trajectories, and compute for the
released, targeted, random, and original-distribution arms. The relative results will tell us
whether the diagnosed examples were a better use of the data budget.

## Confirmation on unseen states

## Targeted data versus random and original-distribution data

## Dose response and regression

## Compute cost

## Limitations

## What should run next

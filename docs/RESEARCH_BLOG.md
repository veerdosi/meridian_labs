# I Gave Codex a Robot Policy and Asked It to Find the Right Training Data

I am trying to answer a fairly practical robotics question: when a capable policy fails, can an AI
research agent work out what went wrong, collect the right corrective data, and prove that its data
choice was better than simply adding more examples?

The robot policy is π0.5. It sees a language instruction, an observer-camera image, a wrist-camera
image, and robot state, then controls a simulated Franka Panda arm in LIBERO. I set the research
objective and the standard of evidence. Around that brief, Codex built and operated a harness that
could choose experiments, read videos and MuJoCo traces, reject bad hypotheses, select training
data, run controlled fine-tunes, and stop when the evidence no longer justified more compute.

## The research loop I wanted

Most robot-policy evaluations end with a success rate. That number is useful for comparison and
leaves the next-data question unanswered. A failed placement can begin with a perception error, a
bad approach, a missed grasp, unstable contact, the wrong strategy, or a correct action sequence
that terminates too early. Each diagnosis implies a different training set.

I gave Codex three questions to answer:

1. Where in the behavior is the gap?
2. What kind of experience is missing?
3. How much corrective data is enough?

I kept the project-level decisions: what claim mattered, what counted as a valid experiment, and
when the evidence required a major pivot. I did not prescribe each rollout or debugging step. Codex
owned the work inside those constraints. It proposed search spaces, chose tasks and states, wrote
the experiment code, submitted compute, diagnosed trajectories, designed controls, selected
demonstrations, and applied the stopping rules. Negative results stayed in the record alongside
positive ones.

The resulting loop looks like this:

```text
run policy
    ↓
record synchronized video and physical state
    ↓
locate a repeatable failure
    ↓
test the diagnosis with unseen states and control settings
    ↓
choose corrective data from a protected training partition
    ↓
fine-tune equal-dose arms
    ↓
evaluate target recovery, regression, and marginal return
```

The harness freezes state partitions, seeds, gates, and comparison rules before Codex inspects
their outcomes. Codex can edit the code that implements the loop, with every change preserved in
version control. Human video review remains part of the protocol because a structurally valid
record can still be a scientifically meaningless experiment.

## The first task and the result I almost kept

For the first campaign, I decided that Codex should look for a boundary in an otherwise competent
policy rather than quietly teach it a task from scratch. Codex selected LIBERO Spatial task 0: pick
up the black bowl between the plate and the ramekin and place it on the plate. The released policy
solved all 40 rollouts in the initial envelope. Codex then widened the search, and 2 of 24 rollouts
failed under a combination of camera translation, camera rotation, image occlusion, visual
distractors, action noise, and faster replanning.

One profile reproduced its failure in four of five new trials. Matched probes then narrowed the
interaction: under the stressed observation, two-step replanning succeeded in 0/5 trials while the
canonical five-step controller succeeded in 3/5. That looked like a useful boundary. The policy
could solve the task, but unstable observations combined with frequent action resampling appeared
to break it.

The first data intervention was informative but negative. Targeted data improved the released
checkpoint from 9/20 to 16/20 target successes, but random data reached 19/20 and
original-distribution data reached 18/20. Codex treated that as a failed hypothesis and revised the
selector rather than relabeling the best arm. Its second, locked dose-eight campaign concentrated
six examples near the measured boundary and kept two for broader coverage.

Every trained arm started from the same released `pi05_libero` checkpoint and fine-tuned LoRA
adapters for 100 optimizer steps. Codex used batch size 2, AdamW with gradient clipping at 1.0, and a
cosine schedule with 10 warmup steps and a peak learning rate of 5 × 10⁻⁵. The π0.5 action horizon
remained 10. The first comparison used 24 trajectories per arm; the refined comparison used eight.
Nothing except the selected training trajectories changed between equal-dose arms.

This time the table looked excellent:

| Condition                  | Target success | Regression success |
| -------------------------- | -------------: | -----------------: |
| Released checkpoint        |          21/40 |              19/20 |
| Evidence-weighted targeted |          40/40 |              20/20 |
| Same-dose random           |          37/40 |              20/20 |
| Original distribution      |           8/40 |              18/20 |

Targeted beat the released checkpoint by 19 trials and random by three, with no target losses in
either paired comparison. It also reached the ceiling, so the stopping rule rejected a larger dose.
Codex's locked gate called the campaign positive. I thought I had the first complete result, but I
asked for a visual audit before accepting it.

Codex found that a black mask covered roughly a quarter of the observer image. The accompanying camera transform
pushed the robot, the manipulated bowl, and the goal plate outside the field of view. Worse, the
policy also received a wrist-camera stream, but the evidence video did not show it. I could not tell
whether the intervention had created a difficult but solvable viewpoint or simply removed the
information required to solve the task.

I decided not to use the result as evidence for the intervention claim.

![The recording preflight with clean observer view, policy input, and wrist-camera view.](../artifacts/physical-preflight/15246052/diagnostic-montage.png)

_The rebuilt recording path. This preflight used canonical observations, so the clean observer view
and policy input match. The wrist image is π0.5's second visual input._

I made synchronized visual evidence a requirement for every later campaign. Codex made that
requirement executable: the harness now checks synchronization, image orientation, finite actions
and states, contact records, object motion, and the task predicate before training can proceed.

## Turning a rollout into physical evidence

Video is excellent at showing that the robot grabbed the wrong thing. It is less useful for
measuring whether an object moved 2 millimetres or 20 centimetres, or whether contact happened just
before the policy changed direction. For that, the rollout recorder saves the simulator state at
every step.

Each trajectory includes:

- the action sent by π0.5;
- robot and end-effector state;
- MuJoCo joint positions and object poses;
- robot-to-scene contacts;
- the LIBERO BDDL goal predicates;
- the clean observer image, policy image, and wrist image; and
- provenance for the task, initial state, checkpoint, and control parameters.

The diagnostic layer turns those measurements into small, checkable claims. It can identify an arm
that moved without reaching contact, contact followed by negligible object motion, a relation that
was achieved and later broken, or a distractor that moved while the task's target stayed still.

The BDDL predicate is especially important. A task such as `On(frying_pan, stove)` tells us which
object is the target and which entity defines success. The harness resolves those symbols to the
corresponding MuJoCo joints, then measures the target and other movable objects separately. This
becomes important later because the first diagnostic implementation got exactly this part wrong.

## A 60-rollout search across LIBERO

After rejecting Task 1, I made the main scientific pivot: stop manufacturing a viewpoint boundary
and search for a natural physical failure in LIBERO. Codex designed a balanced screen over ten
tasks, with two tasks from each of LIBERO Spatial, Object, Goal, LIBERO-10, and LIBERO-90. π0.5 ran
six canonical rollouts per task for 60 trials in total.

Before inference, Codex ran a state-only inventory over 500 initial states and verified that none
already satisfied its goal. It also separated states by purpose before seeing outcomes:

| Partition         | Use                              | State indices      |
| ----------------- | -------------------------------- | ------------------ |
| Discovery         | Find a repeatable behavior       | 0, 3, 6, 9, 12, 15 |
| Confirmation      | Challenge the diagnosis          | 18, 21             |
| Training source   | Select corrective demonstrations | 25 to 39           |
| Untouched holdout | Final target evaluation          | 40 to 49           |

The policy solved 48 of 60 trials. Eight tasks were perfect at 6/6. Two LIBERO-90 tasks were 0/6.

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

The aggregate success rate was 80 percent, but the task split told a better story. π0.5 was not
slightly worse everywhere. It was completely reliable on eight tasks and repeatedly wrong on two.

The initial selector had been built to find pose boundaries within otherwise competent tasks. It
looked for nearby initial states that separated successes from failures. A task with six failures
and no successes therefore received a poor score. I had already ruled out selecting a task merely
because it failed. Codex therefore tested whether these were purposeful, repeatable failures with a
specific mechanism and a reason to expect intervention headroom.

The physical traces showed Codex that the robot was active. Every failed rollout used the 400-step
horizon. The end effector travelled roughly 1.1 to 1.8 metres and spent substantial time in contact
with scene objects. The policy had a strategy. Codex now had to identify which strategy it was
executing.

## The diagnostic bug that hid the behavior

The first diagnosis summarized whichever free object moved the most. That feature is useful when I
only want to know whether the robot manipulated something. It fails on relational tasks because
moving the wrong object can look like progress.

The BDDL goals and the videos disagreed with the summary. Codex traced the mismatch to the object
resolver, changed it to derive the commanded object from the predicate, and reported target and
distractor displacement separately. It also added an integration test built from a real evaluation
record and trajectory archive. The earlier tests used synthetic records and had never exercised the
actual result layout.

Running the corrected diagnosis on the same trajectories changed the interpretation completely.

For **put the frying pan on the stove**, the frying pan was effectively stationary in all six
states. The moka pot moved 19 to 30 centimetres in five trials. It moved 2.43 centimetres in the
sixth, just below the locked 2.5-centimetre threshold, so Codex marked that case ambiguous.

For **put the white bowl to the right of the plate**, the bowl moved less than 1 millimetre in five
trials while the plate moved 3.4 to 22 centimetres. Both objects moved in the remaining trial, which
Codex also marked ambiguous.

Ten of the 12 discovery failures had the strict signature: negligible target motion and substantial
motion of the wrong object.

![Frames from the two discovery tasks, with the requested target and the object the policy moved.](../artifacts/physical-campaign/screening/15246322/representative-videos/failure-montage.png)

_π0.5 moves the moka pot instead of the frying pan, and the plate instead of the white bowl._

The synchronized discovery rollouts are here:

- [Frying pan to stove, state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i0-diagnostic.mp4)
- [Frying pan to stove, state 3](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i3-diagnostic.mp4)
- [White bowl to right of plate, state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i0-diagnostic.mp4)
- [White bowl to right of plate, state 9](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i9-diagnostic.mp4)

At this stage, I had a plausible object-role or strategy-selection failure. I required a locked
confirmation before any training spend. Codex still had to test whether the policy would recover
with a longer episode or different action replanning.

## Trying to make the failure disappear

The confirmation plan used two previously unseen initial states for each task and four control
settings per state:

| Control             | Action replanning interval |   Horizon |
| ------------------- | -------------------------: | --------: |
| Canonical           |                    5 steps | 400 steps |
| Rapid replanning    |                     1 step | 400 steps |
| Longer action chunk |                   10 steps | 400 steps |
| Extended episode    |                    5 steps | 800 steps |

This produced 16 confirmation rollouts. Codex wrote the gate before any outcome was inspected. The
behavior had to repeat on both tasks, appear in at least 12 of 16 traces, and persist under the
replanning and horizon controls. I retained a final requirement that the representative videos also
make physical sense to a human reviewer.

π0.5 succeeded in **0 of 16** trials. The corrected telemetry found the wrong-object signature in
**16 of 16**. Canonical control, rapid replanning, longer chunks, and the doubled horizon all
failed. Human review matched the state traces: the robot manipulated the moka pot instead of the
frying pan and the plate instead of the white bowl.

![Confirmation frames from canonical and extended-horizon trials.](../artifacts/physical-campaign/confirmation/representative-videos/role-confirm-t18-i18-canonical-diagnostic-sheet.png)

_A canonical confirmation rollout for the frying-pan task. The full diagnostic video is linked
below._

Representative confirmation videos:

- [Frying pan, canonical control](../artifacts/physical-campaign/confirmation/representative-videos/role-confirm-t18-i18-canonical-diagnostic.mp4)
- [Frying pan, 800-step horizon](../artifacts/physical-campaign/confirmation/representative-videos/role-confirm-t18-i18-extended-diagnostic.mp4)
- [White bowl, canonical control](../artifacts/physical-campaign/confirmation/representative-videos/role-confirm-t37-i18-canonical-diagnostic.mp4)
- [White bowl, 800-step horizon](../artifacts/physical-campaign/confirmation/representative-videos/role-confirm-t37-i18-extended-diagnostic.mp4)

The confirmation run consumed 43.3244 SU, and the trajectory diagnosis consumed another 0.0411
SU. It was expensive enough to make the next decision meaningful, but much cheaper than training
several interventions around an unstable explanation.

I now have evidence for a narrow claim: on two tasks and unseen initial states, π0.5 repeatedly
executes the motor behavior needed to manipulate an object but assigns that behavior to the wrong
scene entity. The experiment does not yet tell me whether training can repair the behavior. That is
the next test.

## Choosing the data that should fix it

LIBERO provides 50 successful human-teleoperated MuJoCo demonstrations for each selected task.
Human operators supplied the low-level robot actions. Codex diagnoses the behavior, chooses the
episodes, verifies their contents, partitions the data, and converts it for π0.5.

Codex's job begins at selection. It verified the two official source files by hash, checked that the
episodes satisfy the real LIBERO success condition, recovered the target and reference objects from
the BDDL predicates, and kept selection inside the protected training-state pool. It then converted
the selected action, robot-state, observer-camera, and wrist-camera streams into π0.5's training
format.

That conversion needed its own visual check. The raw HDF5 camera arrays require a double flip to
match the canonical orientation used during π0.5 evaluation. Shape checks would not catch an
upside-down image, so the converter verifies orientation as well as dimensions, finite values, and
episode success.

![Raw demonstration orientation compared with the canonical policy orientation.](../artifacts/physical-campaign/data-preflight/hdf5-orientation-audit.png)

_The data preflight that caught the image-orientation mismatch before fine-tuning._

The targeted selector scored the starting geometry of successful episodes, then chose a balanced
set across the two tasks that spreads coverage through that geometry. This is a direct consequence
of the diagnosis: teach π0.5 the target and reference roles across varied initial arrangements while
retaining successful approach, grasp, transport, and placement actions.

## How I judged the intervention

I rejected an easier random control drawn from unrelated tasks because it would confound task
identity with data selection. The primary comparison had to ask whether Codex's diagnosis chose
better examples, not merely whether the model had seen the failing tasks. Codex therefore built
targeted and random sets from the same two task files, the same protected training partition, and
the same dose. Targeted used the diagnosed geometry; random used a fixed seed without geometry
scoring. Original-distribution rehearsal remained a secondary control and would run only if
targeted first beat this stronger same-task random arm.

The campaign started from the released π0.5 checkpoint. The locked sequence allowed four
conditions when the earlier gates justified them:

| Condition             | Training data                                 | Question                                                   |
| --------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| Baseline              | none                                          | How often does the released checkpoint solve the boundary? |
| Targeted              | geometry-selected examples from the two tasks | Did the diagnosis choose useful data?                      |
| Same-task random      | seeded random examples from the same pool     | Would arbitrary examples work as well?                     |
| Original distribution | familiar four-suite examples                  | Does ordinary rehearsal produce the same effect?           |

The initial dose was four episodes in total, two per task. If that dose passed the locked
comparison, the next dose would contain eight episodes in total, four per task, with the smaller
targeted set nested inside the larger one. Nesting mattered because the dose curve needed to measure
the value of adding examples rather than replacing one small sample with an unrelated larger
sample.

The fine-tunes used the same controlled recipe: the released π0.5 checkpoint, LoRA adapters,
100 optimizer steps, batch size 2, AdamW with gradient clipping at 1.0, 10 warmup steps, and a peak
learning rate of 5 × 10⁻⁵. Targeted and random received the same number of parameter updates as well
as the same number of episodes. Only episode selection differed.

Every completed arm was evaluated on 40 target trials from untouched states and 20 regression trials.
The analysis reported per-task and pooled success, paired trial outcomes, Wilson confidence
intervals, and regression against the released checkpoint. I set the decision criterion before
training: targeted must strictly beat same-task random while staying within the predefined
regression limit. Codex would apply that gate without renegotiating it after seeing the scores. If
the gate failed, the intervention would stop at that dose.

Had dose eight run, the marginal-return graph would have covered dose 0, 4, and 8. The useful
quantity was the gain from 0 to 4 compared with the gain from 4 to 8. A small second gain would have
indicated saturation and given the harness a reason to stop collecting data.

## If successful demonstrations did not already exist

This experiment relied on LIBERO's human demonstrations. A future environment may provide a
simulator and a goal definition but no successful trajectories. I would then have Codex build a
task-specific simulator expert before training.

For these manipulation tasks, the expert would use MuJoCo's privileged state:

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

The predicate would serve as the acceptance test. A generated trajectory would enter training only
after it completes the task and remains stable.

## Intervention results

The released checkpoint solved none of the 40 untouched target trials. That was consistent with the
discovery and confirmation runs: on both tasks, π0.5 continued to manipulate the wrong object.
The four geometry-selected demonstrations produced three successful target rollouts. Random
same-task data produced one.

| Condition             | Target success | Wilson 95% interval | Regression success |
| --------------------- | -------------: | ------------------: | -----------------: |
| Released checkpoint   |           0/40 |        0.0% to 8.8% |              20/20 |
| Targeted dose four    |           3/40 |       2.6% to 19.9% |              18/20 |
| Same-task random      |           1/40 |       0.4% to 12.9% |              20/20 |
| Original distribution |   Not released |                     |                    |

The pooled table hides an important split. Neither trained arm solved the frying-pan task in any of
its 20 trials. On the white-bowl task, targeted reached 3/20 and random reached 1/20. The targeted
checkpoint therefore improved only one of the two behaviors that motivated the shared diagnosis.

The paired comparison was also much weaker than the raw ordering suggested. Against random,
targeted had three wins, one loss, and 36 ties. The two-sided exact sign test was 0.625. Against the
released checkpoint, targeted had three wins and no losses, but with only three discordant trials
the corresponding value was 0.25. Numerically, targeted was best. Statistically, the experiment did
not establish a reliable advantage.

Codex inspected the three successful trajectories rather than treating the success bit as enough.
They were real completions: the white bowl moved 31.9 to 36.3 centimetres, the plate remained
effectively stationary, and the actual BDDL predicate became true. But each success occurred only
on repeat one. Repeat zero failed at the same initial state for states 44, 47, and 49. The adaptation
had found a behavior that sometimes worked, not a repeatable repair.

This was still a useful result. The intervention moved the policy in the intended direction, and
the same-task random control showed that episode selection mattered numerically. It was nowhere near
strong enough to support the intervention claim.

## Dose response and saturation

The locked plan allowed a larger dose only after dose four beat the controls without unacceptable
regression. That condition did not hold, so Codex did not train dose eight.

This means there is no honest saturation curve for this campaign. There are only two observed
points: zero added demonstrations and four. Drawing a smooth marginal-return curve through them
would imply evidence that does not exist. The stopping decision is itself the quantity result:
collecting more examples from this selector was not justified.

The result also exposed a mismatch between diagnosis and data prescription. The behavior looked
like object-role assignment, but the targeted selector optimized geometric coverage among existing
successful demonstrations. Every demonstration in both arms already showed the correct object for
the task. Geometry was therefore only an indirect proxy for the information the policy appeared to
be missing. The experiment tested whether geometrically diverse same-task data helped, not whether
contrastive role-binding data repaired the mechanism.

## Regression

The released and random checkpoints solved all 20 regression trials. Targeted solved 18. Both
failures occurred on the LIBERO Object task `pick up the tomato sauce and place it in the basket`,
at initial states 40 and 48.

The failure was not another object-selection error. In both traces, the robot contacted and moved
the commanded tomato sauce. It moved the object by roughly 8 to 9 centimetres but never satisfied
the basket predicate before the 400-step timeout. The evidence extractor classified both as
sequencing or termination failures. Video review agreed: the policy knew which object to manipulate
but failed to finish the placement.

Regression fell from 20/20 to 18/20, an absolute loss of 10 percentage points. The locked limit was
2 points. Because the evaluation had 20 trials, even one additional failure would have exceeded
that limit; targeted produced two.

Codex applied the rule written before training and stopped the campaign. It did not release the
original-distribution arm and did not increase the dose. The intervention experiment was valid, but
the intervention was unsuccessful: a small, unstable target gain came with larger damage to an
existing capability.

## What I would change next

The experiment left two different research questions, and they should not be blurred together.

The first is a task-level repair. The two selected tasks contain a repeatable wrong-object behavior,
and teaching a previously unsolved task is still a legitimate data-coverage intervention. Doing it
properly would require data that directly varies the disputed roles. Codex would build a
privileged-state MuJoCo expert, place target and distractor objects in controlled configurations,
execute verified correct trajectories, and reject every trajectory that does not satisfy the BDDL
goal. Targeted data would concentrate on configurations that elicit the wrong-object behavior;
random data would come from the same generator and task but sample the valid state space uniformly.
A fixed replay set would be shared by every arm to protect existing capabilities.

That experiment would support a precise claim if it succeeded: Codex found a shared object-role
failure, generated corrective simulator experience, and taught π0.5 the two tasks more efficiently
than random data without damaging its other skills. It would not show a narrow failure boundary
inside tasks that π0.5 already solved, because the released policy was 0/40 on the selected target
plan.

The second question is the original boundary-repair claim. Eight tasks in the screen were already
6/6. A compact physical search over those competent tasks could vary object pose, receptacle pose,
target-to-distractor distance, or robot approach configuration while preserving visibility and
reachability. A qualifying task would need a competent canonical side, a repeatable moderate
failure side, and enough headroom for targeted data to improve it. The same simulator expert could
then generate targeted, uniformly random, and original-distribution trajectories from one source,
removing the data-source confound.

That campaign gave me what the stopping rule was meant to provide. The harness found a real
behavior, survived a diagnosis bug, tested competing control explanations, ran a matched
intervention, and rejected an unsafe result. It also separated the next decision cleanly: repair a
task-level role gap, or search for a physical boundary in an otherwise competent task.

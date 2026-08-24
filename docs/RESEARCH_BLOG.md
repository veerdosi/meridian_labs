# Can Codex Improve a Robot Policy End to End?

*Building a research harness that turns rollout evidence into a controlled data intervention*

π0.5 can control a robot. It accepts language, two camera views, and robot state, then predicts
actions. What it cannot do by itself is investigate its own failures. A policy does not decide
which failure deserves study, determine whether an evaluation is scientifically valid, identify
the missing experience, construct matched data controls, or stop an experiment when the evidence
no longer justifies the cost.

That is the system we set out to build around it.

The central idea is to use Codex as the research agent around a separate robot policy. π0.5 remains
the controller inside each rollout. Codex works at the level above it: it builds and audits the
experimental apparatus, reads physical telemetry and visual evidence, forms competing
explanations, locks evaluation plans before seeing outcomes, selects training data, launches
bounded simulator and training runs, and updates the research record as evidence arrives.

This is not a claim that a language model becomes a roboticist merely because it can call a shell.
The harness has to give Codex the right observables, constraints, and tests. It also has to make bad
reasoning expensive to hide. A proposed diagnosis must agree with the trajectory. A target
condition must remain solvable. Training arms must use the same checkpoint, data dose, and
evaluation states. Holdouts must stay untouched. Negative gates must stop spending.

The ultimate test is empirical. Can the intervention selected by Codex improve target-task returns
over the released π0.5 checkpoint, equal-dose random data, and equal-dose data from the policy's
original training distribution, while keeping regression and compute cost within fixed limits?

That comparison is still in progress. This article reports the research loop that has already been
built and the natural policy failure it found. It does not claim the intervention result early.

## The division of labor

The system contains two different kinds of intelligence, and confusing them would make the result
hard to interpret.

**π0.5 is the robot policy.** During a LIBERO rollout, it receives the task instruction, the
canonical observer image, the wrist-camera image, and proprioceptive state. It returns action
chunks that are executed in MuJoCo. It does not inspect aggregate experiment results or modify its
own training set.

**Codex is the research agent.** It does not generate the robot's actions. It operates the loop
around the policy:

1. define a bounded search over tasks and simulator states;
2. validate that the observations and task conditions preserve the information needed to act;
3. inspect videos, goal predicates, contacts, object motion, and robot trajectories;
4. distinguish observable failure stages and maintain competing explanations;
5. lock confirmation thresholds, seeds, partitions, and stopping rules before inspecting the next
   outcomes;
6. choose a data intervention that follows from the evidence;
7. construct matched targeted, random, and original-distribution training arms;
8. orchestrate simulator, conversion, training, and evaluation jobs within explicit cost bounds;
9. compute paired outcomes, uncertainty, regression, and marginal return per added trajectory; and
10. preserve the evidence needed for a human to audit the conclusion.

The human sets the high-level research objective and can challenge a decision. The harness turns
that objective into an executable protocol and gives Codex enough structured evidence to reason
like an ML and robotics experimenter rather than a job scheduler.

The distinction also defines the baseline. "Model alone" means the released π0.5 checkpoint under
canonical control. Any gain from the full system must come from a better research decision about
data, not from changing the evaluator or quietly making the task easier.

## What the research agent must decide

Our original brief was not "fine-tune π0.5 on a failing task." It asked three linked questions.

### Where is the capability gap?

A task success bit is the final symptom, not the mechanism. A useful diagnosis needs to localize
the observable failure to perception, localization, approach, contact, control, recovery,
sequencing, memory, strategy selection, or termination. The categories are not labels to paste
onto a video. Each needs physical evidence and should retain ambiguity when several explanations
fit.

### What experience is missing?

Possible deficiencies include task, object, environment, viewpoint, initial-state, contact-state,
strategy, recovery, temporal, and embodiment diversity. "More data" is not a useful answer. A data
proposal must follow from the diagnosed behavior and make a prediction that equal-dose controls
can falsify.

### How much is required?

The useful quantity is the smallest data dose that changes the behavior reliably without causing
unacceptable regression. That means measuring marginal improvement and saturation. A successful
small dose should not automatically authorize a larger one, and a failed dose should not trigger
unbounded training.

Together, these decisions create the loop we want Codex to execute:

> measure competence, diagnose a repeatable mechanism, run cheap discriminating controls, select
> data, train matched variants, evaluate target gain and regression, then decide whether another
> dose is justified

The scientific contribution is the closed loop, not any single script inside it.

## A false start taught the harness what to distrust

Our first completed intervention looked encouraging on a score table. We had created a visual
condition, collected targeted trajectories, trained variants, and measured an improvement. The
software path from rollout to fine-tuning worked.

Then Codex performed the visual audit that the initial pipeline should have required.

In representative target rollouts, a black mask removed roughly a quarter of the observer image.
A camera transform also pushed the robot, manipulated bowl, and goal plate outside the observer
field of view. Worse, the policy received a wrist-camera stream that the old evidence video had
not recorded. We could not verify exactly what information the policy retained.

The numerical result was therefore not evidence of the robustness claim we wanted to make. The
condition partly deleted the task rather than revealing a meaningful boundary within it. This was
the most important early lesson: a complete pipeline can still produce an invalid experiment.

We kept that false start as a methodological lesson, not as empirical support. It changed the
harness in three ways.

First, intervention validity became a gate. The robot, manipulated object, receptacle, and goal
region must remain meaningfully observable. A difficult input is useful; an information-destroying
input is not.

Second, every rollout now has to expose the exact inputs used by the policy. An observer render is
not an adequate substitute for the image tensor that drove action prediction.

Third, a deterministic selector became one input to judgment rather than an authority. A frequent
failure is not automatically a good scientific target. The candidate also needs canonical
competence or a clearly defined out-of-distribution comparison, repeatability, intervention
headroom, controlled persistence, a specific coverage hypothesis, and low expected coverage from
random data.

This is where the harness became a research instrument rather than automation around a benchmark.

## Making robot behavior legible to Codex

Codex can only reason from the evidence the runtime records. We therefore hardened each rollout to
save synchronized visual and physical streams.

The visual record contains:

1. the clean, unperturbed observer view;
2. the exact observer image supplied to π0.5;
3. the wrist-camera image supplied to π0.5; and
4. a side-by-side diagnostic video with task parameters and success metadata.

The physical record contains actions, robot state, MuJoCo joint positions, object poses,
robot-scene contacts, and the exact LIBERO BDDL goal predicates at every step. Initial simulator
states, trajectory records, and plans are hashed. State partitions and seeds are versioned before
outcome inspection.

![A three-panel preflight montage showing the clean observer view, exact policy input, and wrist-camera view.](../artifacts/physical-preflight/15246052/diagnostic-montage.png)

*Figure 1. The recording preflight. The clean observer view and policy input are identical because
the current campaign uses canonical images. The wrist view is the second visual stream consumed by
π0.5. This run verified synchronization, finite actions and state, contact telemetry, object
telemetry, and exact task predicates. It was an engineering check, not a scientific result.*

The combination matters. Video lets a researcher judge visibility and intent, but it is an
imprecise contact sensor. Telemetry can show that an object translated 20 centimetres, but it
cannot tell us whether the visual behavior looks coherent. Codex reads both. Any automated
diagnosis has to cite quantities that a person can compare against the synchronized rollout.

The diagnostic layer uses conservative observable patterns:

- end-effector motion without robot-scene contact;
- contact without meaningful object motion;
- meaningful target motion followed by loss of the achieved relation;
- target-object motion versus distractor-object motion; and
- physical progress without satisfaction of the exact goal predicate.

These observations can localize a failure stage. They cannot, by themselves, prove why the model
learned that behavior. A causal data claim requires repetition, controlled probes, and a matched
intervention.

The first analysis of the new traces exposed a subtle implementation error. It summarized the
most-moved free object in each scene. On a relational task, the most-moved object can be exactly the
wrong object. Codex caught the conflict between the summary, the goal predicate, and the video.
The extractor now derives the intended target from the BDDL predicate and measures target and
distractor motion separately. Integration coverage uses real rollout records in addition to
synthetic fixtures, so the research code is tested against the schema it actually analyzes.

This is a small engineering detail with a large scientific consequence. Without target-aware
telemetry, the same trajectory could be misclassified as manipulation progress rather than object
selection failure.

## Searching broadly without searching indefinitely

Once the instrument passed its preflight, Codex ran a bounded screen over 60 natural LIBERO
initial states. The design covered ten tasks with six states per task. Two tasks came from each of
LIBERO Spatial, Object, Goal, LIBERO-10, and LIBERO-90.

All rollouts used canonical observations and canonical π0.5 control. There was no mask, image
post-processing, or camera transform. Before policy inference, a state-only inventory checked 500
predefined initial states, 50 per task, and confirmed that none began with its goal already
satisfied.

The state partitions were fixed before outcomes were inspected:

| Partition | Role in the experiment | State indices |
| --- | --- | --- |
| Discovery | Locate a candidate behavior | 0, 3, 6, 9, 12, 15 |
| Confirmation | Test the selected mechanism under controls | 18 and 21 for each selected task |
| Training source | Candidate demonstration pool | 25 to 39 |
| Untouched holdout | Final target evaluation | 40 to 49 |

The screen and final evaluation answer different questions. Discovery is allowed to select a
hypothesis. The untouched holdout is not. Once the target mechanism and training arms are fixed,
the holdout measures whether the intervention transfers beyond the trajectories used to select
and train it.

π0.5 succeeded on 48 of 60 discovery rollouts. The aggregate 80 percent score was less informative
than the task-level structure:

| Suite | Task | Canonical successes |
| --- | --- | ---: |
| LIBERO Spatial | black bowl between plate and ramekin to plate | 6/6 |
| LIBERO Spatial | black bowl from top drawer to plate | 6/6 |
| LIBERO Object | alphabet soup to basket | 6/6 |
| LIBERO Object | milk to basket | 6/6 |
| LIBERO Goal | open the middle drawer | 6/6 |
| LIBERO Goal | cream cheese to bowl | 6/6 |
| LIBERO-10 | turn on stove, then place moka pot | 6/6 |
| LIBERO-10 | book to back compartment of caddy | 6/6 |
| LIBERO-90 | frying pan to stove | 0/6 |
| LIBERO-90 | white bowl to right of plate | 0/6 |

The initial selector had been designed to find a within-task physical boundary, so it expected a
mixture of successful and failed states inside one task. Under that narrow specification, the two
0/6 tasks did not qualify. They had no local success side from which to infer a pose threshold.

Codex did not lower the threshold to manufacture a winner, and it did not discard the observations
just because they violated the selector's assumptions. It inspected the physical traces and asked
whether the pair represented a different, more useful kind of capability gap.

## The policy knew how to manipulate, but selected the wrong role

The 12 failed episodes were not cases where the robot remained still. Every rollout reached the
full 400-step horizon. The end effector travelled approximately 1.1 to 1.8 metres, and
robot-scene contact occupied substantial portions of the trajectories. The released policy could
approach and manipulate objects in these scenes. It was repeatedly choosing the wrong object.

For **"put the frying pan on the stove,"** the BDDL predicate identifies the frying pan as the
target. Across all six discovery states, the frying pan's measured translation was effectively
zero. In five trials, the moka pot moved beyond the locked 2.5-centimetre diagnostic threshold,
with displacement of approximately 19 to 30 centimetres. In the sixth, it moved 2.43 centimetres.
That trial remains ambiguous rather than being rounded into the clean category.

For **"put the white bowl to the right of the plate,"** the white bowl is the target and the plate
is the reference. In five of six trials, the bowl moved less than one millimetre while the plate
moved approximately 3.4 to 22 centimetres. Both objects moved in the sixth trial, so that episode
also remains an ambiguous variant.

![Representative frames from the two failed tasks, showing the requested target and the object the policy actually manipulated.](../artifacts/physical-campaign/screening/15246322/representative-videos/failure-montage.png)

*Figure 2. Representative canonical failures. The inputs are not visually perturbed. In the
frying-pan task, π0.5 repeatedly interacts with the moka pot. In the bowl-plate task, it moves the
plate instead of treating the plate as the reference object.*

Representative synchronized rollouts:

- [Frying pan to stove, discovery state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i0-diagnostic.mp4)
- [Frying pan to stove, discovery state 3](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i3-diagnostic.mp4)
- [White bowl to right of plate, discovery state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i0-diagnostic.mp4)
- [White bowl to right of plate, discovery state 9](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i9-diagnostic.mp4)

Ten of the 12 failures satisfy the strict shared signature: negligible target motion with
substantial distractor or reference motion. Two remain explicitly ambiguous. We call the candidate
mechanism **object selection or relation-role binding**. This name describes the behavior, not an
unobserved circuit inside the network. π0.5 appears capable of the required low-level interaction,
but it often assigns the instruction's manipulated-object role to the wrong entity.

Several explanations still fit. The language grounding could be weak. The policy could be reusing
a familiar but inappropriate task strategy. The 400-step horizon might be too short for eventual
recovery. The action-chunk replanning interval could prevent correction after an early choice. The
confirmation experiment is designed to discriminate these alternatives before training consumes
substantial compute.

If the wrong-object behavior persists, the data hypothesis becomes precise. The missing coverage
is not generic robot motion and not simply more images of frying pans or bowls. It is successful
experience that distinguishes the manipulated target from a distractor or relational reference,
under the exact task instructions and across both tasks.

## Turning the diagnosis into a falsifiable data intervention

The harness must do more than find an interesting failure. It must test whether its diagnosis
selects better data than simpler policies for spending the same data budget.

The planned comparison starts every arm from the same released π0.5 checkpoint and holds training
settings constant. Only the added demonstrations differ:

| Arm | Added experience | Question it answers |
| --- | --- | --- |
| Released checkpoint | None | What return does π0.5 achieve without intervention? |
| Targeted | Balanced successful demonstrations from the two selected tasks | Does data chosen from the diagnosed gap repair the behavior efficiently? |
| Random | The same number of randomly selected demonstrations | Would an arbitrary equal-sized dataset produce the same gain? |
| Original distribution | The same number of familiar π0.5-style LIBERO demonstrations | Is the effect merely additional in-distribution fine-tuning? |

### Where the demonstrations come from

Codex does not invent the low-level action sequence, teleoperate the robot, or autonomously solve a
failed task to create a label. Its contribution is deciding what successful experience the policy
needs and making that decision testable.

The candidate targeted trajectories come from LIBERO's official human-teleoperated demonstrations.
In those records, a human operator controlled the simulated Panda in MuJoCo and completed the task
successfully. For this intervention, Codex diagnosed the shared wrong-object behavior and specified
balanced exact-task coverage: successful frying-pan-to-stove and bowl-to-the-right-of-plate
trajectories, with equal representation from both tasks. Training sources are kept separate from
the predefined confirmation and untouched holdout states, so the final evaluation cannot reuse the
episodes that supplied the corrective behavior.

The harness then handles the data contract around those demonstrations. It validates task identity,
success, trajectory integrity, camera and robot-state streams, and action shape. It records source
provenance and hashes, then converts the selected records into the observation, language, action,
and episode structure expected by the π0.5 training stack. This conversion is checked before a
full training allocation is allowed. A valid demonstration is therefore an existing successful
robot trajectory chosen for a reason, not an action trace authored by Codex.

The two control arms differ only in how the same demonstration budget is spent. The random arm
will draw the same number of successful trajectories from a locked eligible comparison pool,
without using the diagnosed object-role criterion. The original-distribution arm will draw the
same number from the familiar LIBERO task distribution used by the released policy. Their exact
records and hashes belong in the locked data manifest once selection is complete, so we do not
name files before that decision exists.

This is what separates a test of **data value** from generic fine-tuning. All trained arms begin at
the same checkpoint and use the same number of demonstrations, optimizer schedule, training
budget, and evaluation states. If the targeted arm performs better, the distinguishing variable is
the information selected from the diagnosis. If all three improve similarly, the evidence favors
ordinary additional fine-tuning rather than intelligent data selection.

The target arm is balanced across the frying-pan and bowl-plate tasks so that one task cannot
dominate the update. Dose 4 is evaluated first. Dose 8 is conditional on the locked dose-4 gate,
including target gain and regression. This provides a minimal marginal-return curve instead of
assuming that twice the data must be better.

The final target evaluation uses 40 untouched trials, balanced across the two tasks. A separate
20-trial regression plan measures whether specialization damages established capabilities. The
analysis uses paired outcomes where the arms share initial states, Wilson intervals for binomial
success rates, and actual training plus evaluation cost. The stopping decision considers both
absolute target return and the improvement obtained per additional demonstration.

This is the point of putting Codex inside a constrained harness. Codex can infer a data strategy
from rich evidence, but the experiment does not accept that strategy on authority. It compares the
strategy against alternatives that could explain the same gain. If targeted data does not beat the
controls, the correct conclusion is that the diagnosis did not produce a superior intervention.

## Confirmation on unseen states

## Targeted data versus random and original-distribution data

## Dose response and regression

## Compute cost

## Limitations

## What should run next

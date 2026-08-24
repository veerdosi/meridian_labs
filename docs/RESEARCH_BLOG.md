# Where Does π0.5 Break—and What Data Would Actually Fix It?

*A capability-cartography experiment in LIBERO and MuJoCo*

Robot-policy evaluations usually end with a score. We wanted the next decision.

If a capable policy fails, where in the behavior did it fail? What experience is missing? Would a
small, targeted dataset help more than the same amount of generic data? And when should we stop
collecting because the marginal value has disappeared?

This project is an attempt to make those questions executable. The subject is Physical
Intelligence's released `pi05_libero` π0.5 checkpoint. The environment is LIBERO running in
MuJoCo. The research system around them—Meridian—is intended to act as a **capability cartographer
and intervention scientist**: find a real boundary of competence, gather evidence about the
mechanism, prescribe a controlled intervention, and measure whether that prescription was more
valuable than obvious alternatives.

The important word is *real*. We did not want to plant a toy failure, choose a task because it
looked dramatic, or call any post-training improvement a success. The experiment had to begin with
the released policy, discover its own failure, and compare equal-dose data interventions from the
same starting checkpoint.

This article is a living research record. It currently covers the completed search and diagnosis.
Later sections will be filled only as their preregistered experiments finish.

## The three decisions we wanted the system to make

The original idea was broader than a fine-tuning script. We wanted the system to answer three
linked questions.

### 1. Where is the gap?

A task-level failure is only the outer symptom. The useful location might be perception,
localization, approach, contact, control, recovery, sequencing, memory, strategy selection, or
termination. A system that cannot distinguish these stages cannot choose an intelligent next
experiment.

### 2. What kind of data is missing?

The answer might concern task diversity, object diversity, environment or viewpoint coverage,
initial state, contact state, strategy, recovery, temporal structure, or embodiment. Crucially,
"more data" is not a diagnosis. The proposed data must correspond to a falsifiable account of the
observed behavior.

### 3. How much is required?

The useful quantity is not the largest affordable dataset. It is the smallest dose that produces a
credible marginal improvement without unacceptable regression. That requires a dose curve and a
stopping rule, not an assumption that every additional trajectory is valuable.

These decisions define the scientific loop:

> map competence → diagnose a repeatable mechanism → test cheap alternatives → select data →
> train matched variants → evaluate target gain, regression, uncertainty, and cost → decide
> whether to continue

The released policy is the experimental subject. Meridian does not control the robot itself; it
organizes evidence and experiments around the policy.

## The false start that changed the experiment

Our first completed intervention pilot appeared encouraging numerically. We found a visual and
viewpoint condition, generated targeted trajectories, trained controlled variants, and measured
improvement. The pipeline worked end to end.

Then we audited the videos properly.

In representative target rollouts, a black mask removed roughly a quarter of the observer image
while a camera transform pushed the robot, manipulated bowl, and goal plate outside the observer
field of view. The policy also received a wrist-camera stream that the legacy evidence video had
not recorded. We therefore could not tell a clean story about robustness to a difficult but valid
viewpoint. The intervention had partly removed information required to understand the task, and the
recording contract did not expose every image the policy used.

That result remains useful as an engineering lesson, but not as the final scientific claim. A high
post-training score is not sufficient when the target condition itself is poorly defined.

The audit forced three changes.

First, a perturbation must preserve the information required to perform the task. The robot,
manipulated object, receptacle, and goal region must remain meaningfully observable. A failure caused
by deleting the task is not an interesting robustness boundary.

Second, research evidence must show the **exact policy input**, not a convenient observer-only
render. Without it, a reviewer cannot inspect what the model actually saw.

Third, the selector cannot replace scientific judgment. Earlier semantic and visual searches
produced failures, but failure frequency alone does not make a useful intervention target. A
candidate needs repeatability, a specific behavioral mechanism, room to improve, cheap controls
that rule out configuration problems, and an intervention whose coverage differs meaningfully from
random data.

This was the point where the project stopped asking, "Can the fine-tuning pipeline produce a
gain?" and started asking, "Can the evidence support the reason we claim it gained?"

## Building a measuring instrument we could trust

The replacement harness records four synchronized visual streams for every rollout:

1. a clean, unperturbed observer view;
2. the exact image supplied to the policy;
3. the wrist-camera view; and
4. an annotated side-by-side diagnostic video with the rollout parameters and final outcome.

It also records the action stream, robot state, MuJoCo joint positions, object poses, robot–scene
contacts, and the exact LIBERO BDDL goal predicates at every step. Initial simulator states, plans,
and trajectories are hashed. Seeds and state partitions are fixed before outcomes are inspected.

![A three-panel preflight montage showing the clean observer view, exact policy input, and wrist-camera view.](../artifacts/physical-preflight/15246052/diagnostic-montage.png)

*Figure 1. Instrument preflight. The clean observer and policy-input panels are identical because
this campaign uses canonical images rather than a synthetic visual intervention; the wrist view
supplies the policy's second visual stream. The preflight verified synchronized images, finite
actions and state, contacts, object telemetry, and exact task predicates. It was an engineering
validation, not a scientific rollout.*

This distinction matters. Video is excellent for human interpretation, but pixels alone are a poor
way to determine whether a robot touched an object, moved it, completed a subgoal, or later undid
that subgoal. Conversely, telemetry can tell us that an object moved by 20 centimetres but not
whether the resulting behavior looks intentional. The harness keeps both and requires the
diagnosis to cite its physical evidence.

The stage diagnosis is deliberately conservative. It can identify observable patterns such as:

- substantial end-effector motion with no contact or object motion;
- contact without meaningful manipulation;
- a completed subgoal that later regresses;
- target-object motion versus distractor-object motion; or
- physical progress without satisfaction of the task predicate.

It cannot prove from one trace that a training-data gap caused the behavior. That stronger claim
requires repetition, controlled probes, and an intervention test.

One bug in the first telemetry analysis made this rule concrete. The diagnosis initially summarized
the most-moved free object in the scene. On a failed relational task, however, the most-moved object
may be precisely the wrong object. The analysis now derives the intended target from the BDDL goal
predicate and reports target and distractor motion separately. Real rollout records—not only
synthetic unit fixtures—protect that behavior in integration coverage.

## A balanced search instead of a hand-picked failure

With the instrument validated, we screened 60 natural LIBERO initial states: ten tasks, six states
per task. The task set was balanced across five suites, with two tasks from each of LIBERO Spatial,
Object, Goal, LIBERO-10, and LIBERO-90.

The screen used canonical observations and canonical π0.5 control. There was no mask, image
post-processing, or camera transform. Before inference, a state-only inventory checked 500
predefined initial states—50 per task—and verified that none began with its goal already satisfied.

The experiment partitioned states before looking at outcomes:

| Partition | Purpose | State indices |
| --- | --- | --- |
| Discovery | Find candidate behavior | 0, 3, 6, 9, 12, 15 |
| Confirmation | Test the candidate under controls | 18, 21 for the selected two-task hypothesis |
| Training source | Candidate demonstration pool | 25–39 |
| Untouched holdout | Final target evaluation | 40–49 |

The holdout states are not an extension of the screen. They remain untouched until the boundary
rule and data intervention are fixed.

The policy completed 48 of the 60 rollouts. More revealingly, the result was sharply divided by
task:

| Suite | Task | Successes |
| --- | --- | ---: |
| LIBERO Spatial | black bowl between plate and ramekin → plate | 6/6 |
| LIBERO Spatial | black bowl from top drawer → plate | 6/6 |
| LIBERO Object | alphabet soup → basket | 6/6 |
| LIBERO Object | milk → basket | 6/6 |
| LIBERO Goal | open the middle drawer | 6/6 |
| LIBERO Goal | cream cheese → bowl | 6/6 |
| LIBERO-10 | turn on stove, then place moka pot | 6/6 |
| LIBERO-10 | book → back compartment of caddy | 6/6 |
| LIBERO-90 | frying pan → stove | 0/6 |
| LIBERO-90 | white bowl → right of plate | 0/6 |

The original physical-boundary selector expected a mix of successes and failures within one task.
By that narrow rule, neither 0/6 task qualified: there was no local success side from which to infer
a pose threshold. Blindly following the selector would have thrown both tasks away.

That would also have thrown away the strongest pattern in the data. The right scientific response
was not to lower the gate until something passed. It was to ask whether the two tasks exposed a
different kind of gap than the search initially expected.

## The robot acted—but on the wrong object

The 12 failures were not inert episodes. In every one, the policy ran for the full 400-step horizon.
The end effector travelled roughly 1.1 to 1.8 metres, and robot–scene contact occupied substantial
fractions of the trajectories. The robot was doing something; the question was what.

The answer was strikingly consistent across two different relational tasks.

For **"put the frying pan on the stove,"** the BDDL predicate names the frying pan as the target.
Across all six discovery states, its measured translation was effectively zero. In five trials the
moka pot—the other movable object in the scene—moved well beyond the 2.5-centimetre diagnostic
threshold, reaching approximately 19 to 30 centimetres. The remaining trial moved it 2.43
centimetres, just below the locked threshold, and is treated as ambiguous rather than rounded into
a clean result.

For **"put the white bowl to the right of the plate,"** the white bowl is the target and the plate is
the reference object. In five of six trials, the bowl moved less than one millimetre while the plate
moved roughly 3.4 to 22 centimetres. In the sixth, both objects moved, so that episode is also kept
as an ambiguous variant.

![Representative frames from the two failed tasks, showing the requested target and the object the policy actually manipulated.](../artifacts/physical-campaign/screening/15246322/representative-videos/failure-montage.png)

*Figure 2. Representative canonical failures. The observations are not visually perturbed. In the
frying-pan task, the policy repeatedly interacts with the moka pot; in the bowl–plate relation, it
repeatedly moves the plate rather than treating it as the reference.*

Representative synchronized rollouts:

- [Frying pan → stove, discovery state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i0-diagnostic.mp4)
- [Frying pan → stove, discovery state 3](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t18-i3-diagnostic.mp4)
- [White bowl → right of plate, discovery state 0](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i0-diagnostic.mp4)
- [White bowl → right of plate, discovery state 9](../artifacts/physical-campaign/screening/15246322/representative-videos/screen-libero_90-t37-i9-diagnostic.mp4)

Ten of the 12 failures therefore satisfy the same strict telemetry signature: negligible target
motion and substantial distractor motion. The other two remain visible in the record as ambiguous
variants. We describe the shared candidate mechanism as **object selection or relation-role
binding**. In plain language, π0.5 appears able to approach, contact, and move objects in these
scenes, but often assigns the instruction's roles to the wrong object.

That phrase is a hypothesis, not a conclusion about the model's internals. The observable fact is
that the requested target stays still while a distractor or reference moves. Plausible explanations
include language-role grounding, reuse of a familiar but inappropriate manipulation strategy,
insufficient execution horizon, or sensitivity to the policy's replanning interval. Confirmation
must distinguish those alternatives before any training data is spent on this boundary.

This discovery also sharpens the original data question. If the pattern survives controls, the
candidate missing coverage is not "more robot motion" or "more frying-pan images." It is successful
experience that contrasts **which object is the target, which is the reference or distractor, and
which relation the instruction requires** across both tasks. The eventual intervention must test
that specific account against equal-dose random and original-distribution data. Otherwise an
improvement could simply be generic fine-tuning.

## Confirmation on unseen states

## Targeted data versus random and original-distribution data

## Dose response and regression

## Compute cost

## Limitations

## What should run next

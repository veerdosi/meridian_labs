# PI0.5 LIBERO visual-validity audit (v1)

Date: 2026-08-24

## Decision

The Task 2 locked target is **visually invalid as a viewpoint robustness boundary**. No
training may start without a new scientific decision. The replay loader itself passed its
real cross-job one-trajectory preflight, but that engineering result does not validate the
camera transform.

The audit was performed only after the baseline job had completed. No partial outcomes were
inspected, and no locked plan, seed, threshold, or manifest was changed.

## Task 1: LIBERO Spatial bowl-to-plate

Representative records: canonical probe
`exp_pi05_libero_spatial_boundary_v1-batch3000-episode0002-probe-canonical`, baseline failure
`target-seed8103-repeat0`, and targeted-checkpoint success `target-seed8201-repeat0`.

- Canonical: the robot, manipulated bowl, plate receptacle, and placement region remain in
  view. Occlusion is 0%. Critical content is not clipped.
- Baseline target failure: the programmed black square covers 17,689/65,536 pixels, or
  26.99% of every observer frame (requested `occlusion=0.273879`; integer rasterization),
  plus three small color distractors. With `camera_x=0.08055` and yaw `33.261 deg`, the
  robot and manipulated bowl are outside the observer field of view in the sampled frames;
  the plate is also absent. The cabinet remains visible but is not the goal receptacle.
- Targeted-checkpoint success: the black square covers 15,876/65,536 pixels, or 24.22%
  (requested `occlusion=0.244327`), plus four distractors. With `camera_x=0.10654` and yaw
  `33.966 deg`, the robot, bowl, and plate are again outside the recorded observer view.
- Interpretation: this is not clean evidence of robustness to a meaningful partial
  viewpoint. The observer stream combines large random information deletion with a camera
  transform that removes the task actors. The policy also received a wrist stream, which
  the legacy MP4 did not record; consequently the successful outcome cannot be attributed
  from the observer-only video. Task 1's quantitative result remains an executed result,
  but the visual mechanism claim needs re-audit with synchronized streams.

## Task 2: LIBERO-10 task 3, bowl into bottom drawer and close

Representative completed baseline records: canonical success
`primary-regression-task3-seed62000-repeat0`, target failure
`primary-target-seed62000-repeat0`, and target success
`primary-target-seed62000-repeat1`.

- Canonical: robot, bowl, cabinet/drawer receptacle, and the drawer goal region remain
  visible through the manipulation. Occlusion is 0%.
- Target failure: occlusion is also 0%; the problem is framing. At `camera_x=0.11328` and
  yaw `35.928 deg`, the robot and cabinet/drawer are outside the observer field of view for
  essentially the full 520-step rollout. Only part of the bowl appears at the left edge.
- Target success: at `camera_x=0.09438` and yaw `33.638 deg`, the same critical observer
  content is outside the field of view despite the rollout succeeding.
- Wrist stream: synchronized trajectory frames show the bowl and drawer locally during
  approach/manipulation, but later become mostly tabletop. Thus the policy was not deprived
  of every visual cue, yet the purported observer-view intervention removes the global task
  scene and goal context instead of presenting a difficult but valid alternate viewpoint.
- Interpretation: this is an information-removing camera failure, not a defensible
  robustness boundary. A 2/40 observer-target success rate cannot justify adaptation spend
  until the target is redefined to keep robot, object, receptacle, and goal visible.

## Evidence and recording disposition

The local human-review montage is
`artifacts/visual-audit/representative-rollouts-montage.mp4`. The two Task 2 legacy
diagnostics add synchronized wrist video and explicitly label the canonical panel as a
separate paired-initialization reference. They are useful audit evidence, not substitutes
for a synchronized clean observer stream.

Future `openpi_libero_rollout.py` and replay rollouts now write synchronized
`clean_observer.mp4`, `policy_input.mp4`, `wrist.mp4`, and an annotated `diagnostic.mp4`.
Their trajectory archives retain `image` as the exact policy input and also record explicit
`clean_observer_image`, `policy_image`, and `wrist_image` arrays. The diagnostic metadata
includes the perturbation parameters and final success/failure state.

## Required scientific decision

Keep the existing locked campaign frozen, or version a replacement target with a
visibility-preserving camera envelope and rerun only the baseline/validation screen before
any training. The current locked target must not proceed directly to dose-8 training.

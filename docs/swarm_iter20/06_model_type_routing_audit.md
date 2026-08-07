# Swarm Iteration 20 — Model-Type Routing Audit

## Finding

The trainer advertised `graph_joint_relation` and
`epipolar_bias_v2_lite_pp`, but neither value had an explicit construction
branch. Both therefore entered the final `else` and instantiated
`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.

The lite model import was additionally wrapped in `try/except`, while its file
was absent from the integrated branch. Because no lite branch existed, that
missing class was silently irrelevant. On the committed audit-branch trainer,
executing either launcher would construct the default temporal model.

## Fix

- `graph_joint_relation` now constructs
  `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph`.
- A lite implementation based on the later and simpler branch commit `e639129`
  is added at the path already referenced by the trainer. It also fixes the
  historical single-frame raw-output shape. All early ST layers remain
  standard and only the final layer becomes
  `EpipolarBiasedTransformerEncoderLayer`.
- `epipolar_bias_v2_lite_pp` now explicitly constructs that class.
- Both identities are registered for standalone evaluation.
- The ensemble wrapper again lists only homogeneous Bayesian Tri v2
  checkpoints; a lite checkpoint is not partially loaded into a Bayesian
  builder with `strict=False`.
- The two non-Bayesian launchers no longer print an unused
  `--epipolar_loss_weight` argument.

## Historical evidence boundary

The 2026-08-07 state audit proves that the two full-run commands were started,
but it does not contain their final logs, checkpoint state dictionaries, exact
process HEAD, or a runtime `type(model)` record. The committed audit-branch
trainer routes both names to the temporal fallback, and its lite model file is
absent. Therefore old same-named artifacts are not valid graph/lite evidence
without separate artifact inspection; filenames and printed `model_type` are
insufficient.

The corrected graph route matches its existing full-run launcher, which does
not request raw reprojection. That graph class still has no raw-3D output
contract, so this audit does not claim support for `--return_raw_3d` or
`--reproj_raw_weight > 0` on `graph_joint_relation`.

The published `8.61 mm` and `8.35 mm` anchors remain the documented
stabilized-plus-aug Bayesian Tri v2 ensembles. They are not reclassified as
lite/graph results. The old P01 label claiming a four-checkpoint `8.61 mm`
ensemble is corrected to the two Bayesian checkpoints recorded in the
experiment log.

## Validation boundary

This iteration uses source routing inspection and Python syntax compilation
only. The local checkout has no Torch environment, so no Torch forward test is
claimed. No GPU experiment was run.

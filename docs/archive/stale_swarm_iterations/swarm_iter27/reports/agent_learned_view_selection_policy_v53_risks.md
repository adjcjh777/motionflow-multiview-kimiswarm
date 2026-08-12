# v53 Learned View Selection Policy — Risks and Mitigations

## 1. Gumbel-Softmax Training Instability

**Risk.**  The straight-through Gumbel-softmax estimator can produce extremely
sharp or near-uniform masks when the temperature `v53_lvsp_gumbel_temperature`
is too low or too high.  At low temperature gradients are noisy and may cause
selection collapse; at high temperature the mask is too soft to enforce a genuine
subset selection.

**Mitigation.**  Start with a moderate temperature (τ = 0.5) and anneal it over
the first epoch from 1.0 to 0.3.  Use a minimum-entropy regularizer only as a
secondary loss; monitor the entropy of `α` in TensorBoard and abort runs where
entropy collapses to near-zero within the first 100 steps.

## 2. Collapse to a Fixed View Subset

**Risk.**  The policy may learn to always select the same two views (e.g. front
and side) because they are on average the most reliable.  This reduces the
effective multi-view baseline and can hurt joints that are best observed by the
discarded views.

**Mitigation.**  Add a diversity loss that penalizes low variance of selection
probabilities across joints and across the batch.  Cap the maximum selection
rate (`v53_lvsp_max_views`) and use per-joint rather than per-view selection so
that different joints can exploit different views.  Include an ablation that
runs the policy with `v53_lvsp_min_views = V` to ensure the baseline does not
improve simply from dropping bad cameras globally.

## 3. Redundancy with v52 Uncertainty-Weighted Triangulation

**Risk.**  v52 already predicts per-view/per-joint precision weights and
re-triangulates with them.  If the learned view mask `α` is multiplied by those
weights, v53 may only re-discover what v52 already encoded, yielding no
additional MPJPE gain while adding parameters and latency.

**Mitigation.**  Make the score network explicitly use *residual* information
that v52 does not model: ray angles, camera baseline, and joint-specific
occlusion cues from reprojection residuals.  Evaluate an identity-at-init test:
with `v53_lvsp_identity_gate_init = -6`, a loaded v52 checkpoint should give the
same `val_MPJPE` before and after enabling v53.  Only proceed if the sparse-view
`MPJPE@2/3` improves by more than 0.5 mm.

## 4. Sparse-View Minimum-View Violation

**Risk.**  The Gumbel sample plus top-k masking may, due to the straight-through
estimator, accidentally keep fewer than `v53_lvsp_min_views` effective views in the
forward pass, leading to ill-posed DLT and NaN pose estimates.

**Mitigation.**  Clamp the soft mask during the forward pass to guarantee a
minimum mass: after top-k selection, re-normalize so that the sum over views is
at least `v53_lvsp_min_views / V`.  In the loss, add a hard penalty for any
batch element whose selected mass falls below the threshold.  Use the same
`min_visible_views` check already present in `OmniMultiViewFusionV5`.

## 5. Integration and Checkpoint Compatibility

**Risk.**  Adding v53 inside `OmniMultiViewFusionV5` changes the module's state
and the forward ordering relative to v47/v48/v50/v51 modules.  A loaded v52
or v51 checkpoint may no longer reproduce the baseline if the new gate is not
initialized to identity, or if auxiliary losses are accidentally applied
during validation.

**Mitigation.**  Initialize the residual gate to `-6.0` so the residual branch is
closed at start, and zero-initialize the final score MLP layer so the initial
`α` is uniform (≈0.5).  Gate the auxiliary loss on
`self.training` and `self.epoch >= v53_lvsp_warmup_epochs`.  Add a unit test that
loads a v52 smoke checkpoint, enables v53, and asserts the validation MPJPE
delta is below 0.1 mm before training begins.

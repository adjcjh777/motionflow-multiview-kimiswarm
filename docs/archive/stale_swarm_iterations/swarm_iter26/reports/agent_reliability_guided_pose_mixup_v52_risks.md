# Agent Risk Report: v52 Reliability-Guided Pose Mixup

## 1. Gumbel sampling noise destabilises early training

**Risk.** The candidate generator uses Gumbel-softmax sampling over views. If the temperature `v52_rgpm_temperature` is too low at the start, the sampled candidates can be dominated by a single noisy view, producing large pose jumps and gradient spikes.

**Mitigation.** Start with `τ = 2.0` and anneal to `0.5` over the first few epochs. Keep the identity candidate as candidate 0 and initialise the output projection `W_o` to zero, so the module is identity regardless of sampling noise before the fusion head has learned. Use straight-through Gumbel-softmax to maintain differentiability.

## 2. Extra transformer capacity overfits small datasets

**Risk.** Adding a 2-layer per-joint transformer and candidate embeddings increases the parameter count by ~80–120 k. With small smoke configs or few training samples, this can easily overfit and raise val_MPJPE.

**Mitigation.** Use `v52_rgpm_dropout = 0.1`, a small hidden dimension (`64`), and freeze the base pose estimator for the first epoch. Set the auxiliary loss weight `v52_rgpm_loss_weight = 0.01` so the module learns gradually. If smoke val_MPJPE rises, reduce `v52_rgpm_num_candidates` to 2.

## 3. Reliability miscalibration from v50/v51 steers mixup wrong

**Risk.** v50/v51 reliability is learned from reprojection and temporal residuals. On unseen domains, reliability can be overconfident for bad views or underconfident for good views, causing the mixup to down-weight informative views.

**Mitigation.** Clamp reliability to `[0.05, 1.0]` before sampling. Make `v52_rgpm_use_domain_conditioning` optional and evaluate with it off first. Use the *refined* CDSVR reliability only after the v51 module has been trained and frozen, or train them jointly with a lower learning rate on the v52 parameters.

## 4. View-subset dropout candidate is correlated with v46 sparse-view training

**Risk.** The Bernoulli view-mask candidate overlaps with v46 sparse-view generalisation. If both are enabled, the model may see duplicated view-dropout augmentations, slowing convergence or making it overly conservative.

**Mitigation.** Disable `v52_rgpm_dropout_prob` when `use_v46_sparse_view_generalization` is active, or use it only for the blending branch. Document the flag interaction in `OmniMultiViewFusionV5` and assert that `v52_rgpm_dropout_prob` defaults to 0.0 whenever v46 is on.

## 5. Candidate generation assumes per-view 3-D proposals are available

**Risk.** The proposed architecture consumes `per_view_proposals (B, T, V, J, 3)` from v25/v45 geometry fusion. If a future refactor changes the geometry-fusion output to a single triangulated pose without per-view proposals, RGPM cannot be wired in without re-engineering.

**Mitigation.** In the implementation, make the candidate generator accept either `per_view_proposals` or compute a proxy by back-projecting `pred_3d` onto each camera ray. Add a unit test that verifies the module works with both full proposals and the fallback proxy, ensuring the integration point is robust.

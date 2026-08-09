# v53 Domain-Conditional Normalization of UWT Outputs — Risks and Mitigations

## 1. Redundancy with the v52 precision MLP

**Risk:** v52 already predicts per-view, per-joint precision weights with its own MLP. Adding a domain-conditional recalibration on top may learn the same corrections, producing little gain while increasing parameter count and training time.

**Mitigation:**
- Initialize the v53 weight MLP to identity (zero final layer) so it must earn its contribution through gradients.
- First smoke-test v53 on a frozen v52 checkpoint; compare `val_MPJPE` and `MPJPE@2/3` against re-training the v52 precision MLP end-to-end.
- If the gains overlap, merge the two MLPs by feeding domain embedding directly into v52's precision MLP rather than keeping a separate v53 block.

## 2. Re-triangulation instability

**Risk:** v53 re-triangulates with corrected weights via `weighted_dlt_triangulate`. If the weight correction pushes too many views below the effective triangulation threshold, the DLT matrix can become ill-conditioned, producing NaN/Inf or large pose outliers.

**Mitigation:**
- Clamp `w'` to `[v53_dcn_min_weight, 1.0]` and use the same DLT damping (`1e-4`) as v52.
- Add a post-hoc check that flags batches with fewer than two effective views and falls back to the v52 pose.
- Smoke-test with `v53_dcn_weight_loss_weight = 0.1` first to heavily regularize `Δlog_w`, then relax once stability is confirmed.

## 3. Overfitting on small domains

**Risk:** AIST++ and 3DPW actual have far fewer training clips than H36M/MPI. A per-domain pose affine can overfit to these small domains, memorizing domain-specific mean poses instead of learning a useful normalization.

**Mitigation:**
- Share affine parameters across kinematically similar joint groups (e.g., arms, legs, torso) using `v53_dcn_num_groups < J`.
- Apply weight decay directly to the v53 domain embedding and MLP parameters.
- Evaluate per-domain `MPJPE` in smoke tests; if any single domain regresses by more than 1 mm, increase `v53_dcn_pose_loss_weight` or reduce `v53_dcn_hidden`.

## 4. Warm-start drift from non-identical re-triangulation

**Risk:** Identity-at-init is only useful if `weighted_dlt_triangulate` with `w' = uwt_weights` reproduces the exact v52 pose. Numerical differences (damping, masking, half-precision) can break this guarantee and silently shift the baseline.

**Mitigation:**
- Add a unit test that feeds the same weights through v52 and the v53 no-op path and asserts `|P_out - pred_3d_v52| < 1e-4`.
- Run a forward-only smoke test on a trained v52 checkpoint with `use_v53_domain_conditional_normalization=True` and all gradients disabled; the reported `val_MPJPE` should match the parent within 0.1 mm.
- Store the v52 output as a fallback path when `v53_dcn_identity_init` is true but gradients are off.

## 5. Extra latency and memory in the forward pass

**Risk:** The additional MLPs and a second DLT triangulation add compute and memory. On the RTX 4090 smoke config, this could push batch sizes down or step time up.

**Mitigation:**
- Keep MLPs shallow (2 layers, hidden 64) and skip the re-triangulation when `v53_dcn_weight_loss_weight = 0` and the weight correction is zero in early warmup.
- Profile one epoch before committing to the A800 queue; if step time increases by more than 5%, consider caching the v52 DLT result or applying v53 only every other training step.

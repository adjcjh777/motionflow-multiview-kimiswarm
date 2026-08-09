# v54 Self-Supervised Multi-View Pretraining — Risk Register

**Agent:** design-swarm
**Module:** `self_supervised_multiview_pretraining_v54`

## R1. Compute overhead from masked-view triangulation

**Risk.** The masked-view triangulation (MVT) loss performs an extra DLT/weighted-triangulation step for every training sample. On the full A800 batch size this can add 10–20% step-time overhead, especially because it must run inside the training graph.

**Mitigation.**
- Apply MVT stochastically: only 50% of training samples and mask at most 25% of views.
- Cache triangulation matrices and reuse the existing v52 `UncertaintyWeightedTriangulationV52` forward path rather than re-implementing DLT.
- Make the loss optional per-epoch via a flag (`v54_ssmvp_mvt_active`) so it can be disabled during the final convergence phase.

## R2. Auxiliary losses dominate or vanish

**Risk.** The self-supervised terms (MVT, CVFC, TPC) may either drown the supervised `MPJPE` loss or be too weak to train the residual gate. Either case stalls v54 or degrades the v53 baseline.

**Mitigation.**
- Initialize the 3D residual gate to `-6.0` (σ ≈ 0) so the supervised path is unchanged at start.
- Use per-loss gradient-norm clipping and a warmup schedule: `v54_ssmvp_warmup_epochs` before the loss is active, then linearly ramp the weights over the next epoch.
- Provide a smoke test that asserts `val_MPJPE` is within 0.1 mm of the v53-only checkpoint when v54 is enabled at init.

## R3. Masked triangulation creates invalid training signals

**Risk.** Masking too many views or masking the only reliable view can produce noisy `X^masked`, making the reprojection target itself unreliable. The model may learn to ignore high-uncertainty views rather than exploit them.

**Mitigation.**
- Enforce `v54_ssmvp_mvt_min_views >= 2` and sample masks with probability proportional to the v52 UWT weights so reliable views are preferentially kept.
- Weight the loss by the product of per-view UWT confidences for the masked views; low-confidence targets contribute less.
- Clamp per-joint reprojection errors and use Huber loss instead of raw L2 to reduce outlier influence.

## R4. Temporal continuity loss couples with v47/v49 temporal blocks

**Risk.** v54 TPC adds another temporal smoothness objective on top of v47/v49 temporal aggregation. The two may conflict: v54 pushes for short-range frame consistency, while v47/v49 already enforce long-range temporal context, leading to over-smoothed or temporally drifted poses.

**Mitigation.**
- Keep TPC local: predict only the next-frame residual, not the full temporal context.
- Set `v54_ssmvp_tc_loss_weight` small (default 0.01) and make it zero when v47 or v49 temporal aggregation is active, unless ablation shows benefit.
- Add a diagnostic metric `val_temporal_jerk` to detect over-smoothing.

## R5. Integration fragility in the v51/v52/v53 chain

**Risk.** v54 must consume the output of v53 and reuse v52 UWT weights. If the module order, tensor shapes, or loss accumulation change, the A800 queue and smoke configs will break, and warm-starting from a v53 checkpoint will silently fail.

**Mitigation.**
- Add v54 strictly after `physical_space_calibration_v53` in `OmniMultiViewFusionV5.forward`; gate with `use_v54_self_supervised_multiview_pretraining`.
- Re-use the existing `view_mask` and `domain_id` plumbing; do not introduce new data-loader dependencies.
- Write a unit test that (a) instantiates v53 + v54, (b) loads a v53 checkpoint, (c) runs a forward pass with v54 enabled, and (d) asserts that the output equals the v53 output to within `1e-5`.

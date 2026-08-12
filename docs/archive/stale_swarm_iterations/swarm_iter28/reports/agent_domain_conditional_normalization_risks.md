# v54 Domain-Conditional Normalization — Risks and Mitigations

## 1. Redundancy with v48 domain FiLM and v53 PSC

**Description:** v48 already applies domain-conditional FiLM to feature tokens before triangulation, and v53 already calibrates the pose against floor and bone physical invariants. Adding another domain-conditional affine on the calibrated pose and weights may duplicate these corrections, yielding negligible gains while increasing parameter count and training time.

**Mitigation:**
- Initialize all v54 MLP final layers to zero so the module starts as the identity and must earn its contribution through gradients.
- Smoke-test v54 on a frozen v53 checkpoint and compare against (a) re-training v48, and (b) re-training v53. If the gains overlap, fold the pose affine into v53's residual MLP or merge the weight normalization into v52's precision MLP.
- Gate the pose and weight branches independently (`v54_dcn_pose_loss_weight` and `v54_dcn_weight_loss_weight`) so ablations can isolate the useful path.

## 2. Per-domain scale collapse

**Description:** The learned pose affine `γ_p` can collapse or explode for a domain if the `tanh` reparameterization is not tight enough or the auxiliary loss weight is too small. A scale far from `1.0` for a small domain can distort bone lengths and raise MPJPE.

**Mitigation:**
- Clamp `γ_p = 1.0 + 0.1 * tanh(γ_p)` so the multiplicative scale stays in `[0.9, 1.1]`, and `β_p = 0.1 * tanh(β_p)` so the additive shift stays bounded.
- Initialize `γ_p` to exactly `1.0` and `β_p` to exactly `0.0` by zero-initing the final MLP layer.
- Add the auxiliary loss `λ_pose * (tanh(γ_p)^2 + tanh(β_p)^2)` with `v54_dcn_pose_loss_weight = 0.01` by default; increase it if smoke tests show scale drift.

## 3. Overfitting on small domains

**Description:** Per-domain affine parameters can overfit to small domains such as AIST++ or 3DPW actual, memorizing domain-specific mean poses rather than learning a useful normalization. This is especially risky when `v54_dcn_num_groups = J` (per-joint affine) with many parameters.

**Mitigation:**
- Default `v54_dcn_num_groups = 1` (global pose affine) and only increase granularity if smoke tests show clear gains.
- Apply weight decay to the domain embedding and all v54 MLP parameters.
- Monitor per-domain `MPJPE` in smoke tests; if any single domain regresses by more than `1 mm`, reduce `v54_dcn_hidden` or increase the auxiliary loss weight.

## 4. Warm-start drift from weight clamping

**Description:** Identity-at-init requires that `weights_dcn == uwt_weights` at startup. The `v54_dcn_min_weight` clamp can change weights that were already below the floor, breaking the warm-start guarantee and silently shifting the baseline.

**Mitigation:**
- Only apply the floor clamp after normalizing, and make it optional via `v54_dcn_min_weight`. Set the default to the same floor used by v52 (`0.05`) so the behavior is consistent.
- Add a unit test that feeds a v52 checkpoint through the v54 no-op path and asserts `||weights_dcn - uwt_weights||_∞ < 1e-4` and `||pred_3d_dcn - pred_3d_psc||_∞ < 1e-4 mm` with gradients disabled.
- Store a fallback path that returns the original v53 outputs when `v54_dcn_identity_init=True` and gradients are off.

## 5. Interaction with downstream residual MLP and temporal heads

**Description:** v54 changes the distribution of the 3-D pose fed into the final residual MLP and the v47/v49 temporal heads. If those downstream layers were trained on the original v53 pose distribution, the sudden domain-normalized input can destabilize training or require many epochs to re-adapt.

**Mitigation:**
- Use a `v54_dcn_warmup_epochs` guard so the v54 loss is not applied and the affine is frozen at identity for the first epoch when warm-starting from a v53 checkpoint.
- Keep the residual MLP and temporal heads trainable while v54 warms up so they can adapt to the normalized pose distribution gradually.
- In smoke tests, compare training from scratch vs. warm-starting from v53; choose whichever yields lower `val_MPJPE` before committing the A800 full run.

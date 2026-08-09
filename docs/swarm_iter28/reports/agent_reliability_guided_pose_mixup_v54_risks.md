# v54 Reliability-Guided Pose Mixup: Risk Register

**Module:** `reliability_guided_pose_mixup_v54`  
**Author:** design-swarm agent  
**Related proposal:** `docs/swarm_iter28/proposals/v54_reliability_guided_pose_mixup_v54.md`

---

## 1. Anchor drift away from plausible human pose

**Risk:** The learned canonical anchor `P_anchor ∈ R^(J,3)` is unconstrained. If the mixing weight `α_j` becomes large, a badly initialized anchor may pull joints into anatomically implausible positions and degrade `MPJPE` despite the identity-at-init gate.

**Mitigation:**
- Initialize `P_anchor` from a dataset mean pose when `v54_rgpm_anchor_init == "mean_pose"`; otherwise zero-initialize and freeze it for the first epoch.
- Clamp `α_j` to `[v54_rgpm_min_alpha, v54_rgpm_max_alpha]` so the anchor can never dominate the calibrated pose.
- Add a weak L2 regularizer on `P_anchor - mean_pose` and on the per-joint scale `s_j` to keep the anchor close to a natural skeleton.

---

## 2. Over-correction when v52 reliability is miscalibrated

**Risk:** The module uses the v52 per-joint reliability to decide which joints to mix. If v52 underestimates uncertainty for an outlier joint, v54 will leave that joint unchanged; if it overestimates uncertainty for a good joint, v54 will pull it toward the anchor and introduce error.

**Mitigation:**
- Use a smoothed reliability estimate: `reliability_j = clamp(max_v w_vj, 0.1, 1.0)` with a small floor, so no joint is treated as completely unreliable without evidence.
- Make `α_j` a function of both reliability **and** the per-joint residual norm, so the module mixes more aggressively only when the calibrated pose is also inconsistent with the 2-D evidence.
- Detach the v52 reliability from the v54 gradient path if `v54_rgpm_use_domain_conditioning` is disabled, preventing v54 from corrupting the already-trained v52 weights.

---

## 3. Identity-at-init leakage

**Risk:** Even with `v54_rgpm_identity_init=True`, the mixing MLP output layer is zero-initialized but the anchor `s_j * P_anchor_j` may be non-zero. If `α_j` is not exactly zero at the first step, the forward pass will differ from the v53 baseline by more than the `0.1 mm` warm-start tolerance.

**Mitigation:**
- Bias the final layer of `MLP_mix` so that `sigmoid(logit) = v54_rgpm_min_alpha` at init (typically `0`).
- Initialize the scalar gate `β = sigmoid(v54_rgpm_residual_gate_init)` to a very small value (`≈ 0.002` for `-6.0`).
- Add a deterministic unit test asserting `||pred_v54 - pred_psc||_∞ < 1e-5` when all identity-initialization flags are active.

---

## 4. Conflict with v53 physical-space calibration losses

**Risk:** v53 already optimizes floor and bone constraints. v54 applies an additional per-joint mixup that may move the pose away from the floor/bone optimum found by v53, increasing physical-space loss or producing foot-skating.

**Mitigation:**
- Keep the v54 output as a small gated residual on top of the v53 pose, not a full replacement, by clamping `v54_rgpm_max_alpha <= 0.5` and using `v54_rgpm_residual_gate_init = -6.0`.
- Re-apply the v28 floor loss and a bone-length loss inside the v54 auxiliary loss so that any anchor pull remains physically consistent.
- Optionally feed the v54-refined pose back into the existing physical loss computation rather than the v53 pose, ensuring the downstream loss sees the updated estimate.

---

## 5. Sparse-view collapse to the anchor

**Risk:** With very few visible views (2–3), the v52 reliability for most joints may be low. The module may then default toward the learned anchor for those joints, collapsing the diverse motion of different subjects into a single mean pose and hurting `MPJPE@2`.

**Mitigation:**
- Gate the total v54 loss by the average number of visible views per sample; reduce `v54_rgpm_loss_weight` automatically when `mean_views < 3`.
- Use per-joint rather than global mixing, and condition `α_j` on the local residual to the 2-D evidence so that the anchor is only used when the multi-view data are genuinely ambiguous.
- Monitor the mean `mixup_alpha` during smoke training; if it exceeds `0.3` for sparse views, lower `v54_rgpm_max_alpha` or freeze the anchor.

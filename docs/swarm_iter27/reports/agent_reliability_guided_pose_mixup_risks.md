# v53 Reliability-Guided Pose Mixup: Risk Register

**Module:** `reliability_guided_pose_mixup_v53`  
**Author:** design-swarm agent  
**Related proposal:** `docs/swarm_iter27/proposals/v53_reliability_guided_pose_mixup.md`

---

## 1. Identity-at-init leakage

**Risk:** Although the mixup multipliers and residual gate are initialized to zero, small numerical differences in the repeated `weighted_dlt_triangulate` calls or floating-point drift when scaling weights by `sigmoid(0) = 0.5` could cause `pred_3d_gn_mix` to differ from `pred_3d_gn_uwt` by more than the `0.1 mm` warm-start tolerance.

**Mitigation:**
- Short-circuit the candidate generation path when `v53_rpm_residual_gate == 0.0` and all candidate multipliers are at their initialized values, returning the input pose directly.
- Add a deterministic unit test that asserts `||pred_3d' - pred_3d||_∞ < 1e-5` when the module is identity-initialized.
- Clamp the predicted mixup multipliers to a safe range and avoid in-place weight modifications that could perturb the DLT solver.

---

## 2. Candidate collapse

**Risk:** The scoring MLP may converge to a trivial solution where all candidates receive the same score, making the mixup ensemble effectively a single estimate. This eliminates the benefit of the multi-hypothesis design and wastes parameters.

**Mitigation:**
- Initialize the per-view perturbations `δ_{m,v}` with small, distinct random values rather than all zeros, while keeping the scalar multipliers `β_m` at zero to preserve approximate identity.
- Use the entropy bonus `L_ent` to encourage a spread of attention weights across candidates.
- Monitor the standard deviation of `mixup_scores` during training; if it collapses below a threshold, restart the candidate perturbations from a small random seed.

---

## 3. Extra compute and memory from candidate ensemble

**Risk:** Running `M` separate weighted DLT triangulations per forward pass multiplies the triangulation cost by `M`. With `v53_rpm_num_candidates=4` and large `T` or `J`, this may cause OOM or slow training compared to the v52 baseline.

**Mitigation:**
- Keep the default number of candidates low (`M=4`) and make it configurable.
- Run the candidate triangulations in a loop rather than materializing all candidates simultaneously when memory is constrained; the loop can be compiled with `torch.compile` if needed.
- Cache and reuse the camera projection matrices across candidates, since only the weights `w_m` differ.

---

## 4. Conflict with v52 consistency loss

**Risk:** v52 already trains its precision weights `w_uwt` with a consistency loss and entropy term. Adding v53's score-based mixup loss on top may create conflicting gradients: v52 pushes weights toward reprojection consistency, while v53's score target may pull the same weights in a different direction.

**Mitigation:**
- Detach `w_uwt` before passing it to v53, so v53 only learns its own mixup multipliers and scoring MLP without back-propagating into v52's precision network.
- Keep `v53_rpm_loss_weight` small (`0.005–0.01`) and apply a warmup (`v53_rpm_warmup_epochs > 0`) so v52 stabilizes before v53 is active.
- Monitor the ratio of v53 loss gradient norm to total loss gradient norm and clip if it exceeds `0.1`.

---

## 5. Sparse-view instability

**Risk:** With only 2–3 views, dropping or re-weighting a single view can dramatically change the triangulation. If the per-view perturbations `δ_{m,v}` learn to down-weight one of the few available views, the candidate poses may become noisy or degenerate, hurting `MPJPE@2`.

**Mitigation:**
- Enforce a minimum effective weight floor `v53_rpm_min_weight` so that no view is ever fully zeroed out by the mixup multipliers.
- Add a sparse-view regularizer that penalizes large variance of mixup multipliers when the number of visible views per joint is small.
- Gate the v53 loss and residual contributions based on the average number of visible views, reducing the module's influence in very sparse settings until it has learned stable behavior.

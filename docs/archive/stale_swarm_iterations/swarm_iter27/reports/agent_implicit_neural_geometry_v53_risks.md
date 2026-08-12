# Agent v53: Implicit Neural Geometry — Risk Report

**Owner:** design-swarm agent v53  
**Module:** `implicit_neural_geometry_v53`  
**Tracking issue:** #193  
**Date:** 2026-08-09  

## 1. Risk: Ray construction becomes numerically unstable for degenerate cameras

**Description:** v53 computes camera centers `o_v = -R_v^T t_v` and world-space ray directions `d_vj = normalize(R_v^T K_v^{-1} [u_vj, 1])`. If a camera matrix is near-singular (`det(K)` close to zero) or if `points_2d` contains padding values, the inverse `K_v^{-1}` can explode or produce invalid directions. This is especially likely in variable-view mode when missing views are zero-padded.

**Evidence:** v25 geometry fusion already normalises `K` before use, but it does not explicitly guard against `det(K) ≈ 0`. v33 ray-conditioned attention added a small `eps=1e-6` to ray norms, suggesting this class of modules needs explicit guards.

**Mitigation:**
- Clamp `det(K)` to `≥ 1e-6` before inversion and add a fallback ray direction `(0, 0, 1)` when the norm is below `1e-6`.
- Apply the existing `view_mask` before any ray computation so padded views never enter the MLP.
- Unit-test the ray builder with deliberately corrupted intrinsics and verify it returns NaN-free outputs.

## 2. Risk: The implicit field collapses to an identity map

**Description:** Because `MLP_δ` and `MLP_α` are zero-initialized and the residual gate starts at 0, the module can simply learn to keep `δ_vj ≈ 0` and `α_vj ≈ 0` throughout training. If the auxiliary loss weight is too small, v53 becomes a no-op; if it is too large, it may over-correct and degrade the strong v52 baseline.

**Evidence:** Every warm-startable module in the v25–v52 series (v33 HMSP, v34 VJGN, v36 UGIGR, v52 UWT) uses zero-initialized residual branches and small loss weights to escape the identity basin. v53 adds a second level of correction on top of v52, compounding the cold-start problem.

**Mitigation:**
- Start with `v53_ing_loss_weight=0.01` and `v53_ing_residual_gate_init=0.0`; do not increase until the smoke shows non-zero `mean(|Δp|)`.
- Monitor the mean ray-attention entropy and mean offset magnitude; both should move away from zero within 500 steps.
- Add a small identity regulariser during the first epoch that penalises large `|Δp|`, then decay it to zero.

## 3. Risk: Ray-alignment loss conflicts with v52 reprojection evidence

**Description:** v53 pulls the refined point onto the camera rays, while v52 UWT pulls it toward the algebraic weighted-DLT solution. In cases where the algebraic solution is already accurate (many views, low noise), forcing the point onto each ray can introduce correlated jitter. In cases where the ray itself is noisy (outlier keypoint), the ray loss can pull the point toward the wrong line.

**Evidence:** v33 ray-conditioned attention observed that ray-biased features help when views are noisy but can hurt when the 2-D detector is already accurate. v53 uses a similar geometric inductive bias, so it inherits the same failure mode.

**Mitigation:**
- Weight the ray-alignment loss by the v52 UWT weights so low-confidence views contribute less.
- Gate the loss with `v53_ing_use_ray_alignment` and ablate it independently.
- Add a small reprojection term `L_reproj` in the v53 auxiliary loss to keep the refined point faithful to the observed 2-D evidence.

## 4. Risk: Compounding corrections with v50 SEFH, v51 CDSVR, and v52 UWT

**Description:** v53 is the fourth refinement head in the v5 stack (SEFH → CDSVR → UWT → ING). Each head adds a residual correction to the pose. Stacking them increases the risk of over-correction, gradient interference, and training instability, especially if several heads are enabled at once.

**Evidence:** v43 adaptive per-node residual showed that adding a second residual gate on top of v36 UGIGR required careful tuning. The v52 UWT smoke already adds a residual MLP after triangulation; v53 adds another residual on top of that.

**Mitigation:**
- Default `v53_ing_residual_gate_init=0.0` and keep `v53_ing_loss_weight` small (0.01).
- Run a mandatory ablation with `use_v53_implicit_neural_geometry=True` vs. `False` while holding v50/v51/v52 fixed.
- Optionally freeze v50/v51/v52 weights for the first epoch so v53 learns on top of a stable baseline.

## 5. Risk: Memory and latency overhead break the A800/AIST batch budget

**Description:** v53 introduces a per-(view, joint) MLP (`MLP_δ`, `MLP_α`) and a ray builder that inverts `K` and normalises a direction for every `(B, T, V, J)` sample. For `B=8, T=9, V=4, J=17`, this adds non-trivial compute on top of v52 UWT. Combined with v47 temporal aggregation and v50/v51, it may push the smoke OOM or force a smaller batch size.

**Evidence:** v52 UWT already adds a precision MLP plus weighted DLT, and A800 runs are typically capped at `clip_len=9` and `batch_size=8`. v53’s additional per-view MLP and ray computations could add 5–10 % memory and latency.

**Mitigation:**
- Default `v53_ing_hidden=64` and `v53_ing_n_layers=2` to keep the MLP small.
- Cache `K^{-1}` per view and reuse it across joints/time where possible.
- Run the smoke at `clip_len=3, B=4` first; only scale to `clip_len=9` after confirming GPU headroom.
- Profile the forward pass with `torch.cuda.memory_summary` before committing to the full A800 run.

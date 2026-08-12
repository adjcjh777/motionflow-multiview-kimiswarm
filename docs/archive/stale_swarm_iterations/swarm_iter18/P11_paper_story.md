# P11: OmniMultiViewFusion — The Next Chapter of the ICRA/CVPR 2027 Paper Story

> Drafted on swarm-iter18-omniview (2026-08-07).  
> Continues the living paper story in `docs/icra_cvpr_2027_paper_story.md`.

---

## 1. Why we need a next chapter

The Bayesian triangulation v2 ensemble has reached **8.35 mm** MPJPE on MPI-INF-3DHP S2/Seq1, comfortably below the ICRA/CVPR 2027 publishable threshold of 8.75 mm.  The remaining gaps are no longer raw accuracy; they are **robustness, generality, and architectural unity**.  Specifically:

| Gap | Current state | Risk for camera-ready |
|-----|-------------|----------------------|
| **Discrete vs. learned view selection** | Hard visibility / adaptive gates are brittle and underperform. | Cannot tell a clean occlusion-robustness story. |
| **Fragmented architecture** | PP correction, visibility, uncertainty, and graph-joint modules have never lived in one model. | Paper reads as an ensemble of tricks rather than a coherent method. |
| **Cross-dataset transfer** | H36M ↔ MPI mixed training fails (~101 mm). | Limits ICRA robot-retargeting claims. |
| **Variable camera rigs** | CamPE works but is not integrated with the strongest residual model. | Cannot claim "works with any camera setup." |
| **Calibration rotation robustness** | rot_0.5° still costs ~8 mm. | Field deployment on moving cameras remains risky. |

**The next paper story is therefore:** unify the strongest inductive biases into a single module, *OmniMultiViewFusion*, that is robust, transferable, and simple enough to ship as the default MotionFlow fusion plugin.

---

## 2. The OmniMultiViewFusion thesis (one sentence)

**A single geometry-first fusion module learns to correct intrinsics, gate occluded views, exchange skeleton-aware information, and weight views by uncertainty — all before triangulating, refining, and adding a final residual correction.**

---

## 3. From Bayesian tri v2 to OmniMultiViewFusion

### 3.1 What we keep

| Component | Role in OmniMultiViewFusion |
|-----------|----------------------------|
| Anisotropic covariance / precision head | Becomes the **uncertainty head** that predicts per-view log-variance `λ_vj`. |
| Adaptive Gauss-Newton step | Kept as the geometric refinement after weighted triangulation. |
| Principal-point / intrinsic correction | Upgraded to a full **intrinsic self-calibration head** that predicts `Δ(cx,cy)` and `Δf` and is supervised by the inverse perturbation. |
| Cross-view spatio-temporal attention | Replaced by a **factorised (T × V × J) Transformer block** that is cheaper and more expressive. |
| Residual refinement MLP | Kept as the final correction, now conditioned on pooled factorised features. |

### 3.2 What we add

1. **Visibility head before fusion.**  A per-view, per-joint soft visibility multiplier `m_vj ∈ [0,1]` is predicted early from raw 2D/confidence and used to mask occluded views in the view-attention layers and in the weighted triangulation.
2. **Graph-constrained joint attention.**  Dense joint self-attention is replaced by a sparse skeleton graph (parent–child, symmetry) inside the factorised block, giving anatomically plausible propagation of evidence.
3. **Uncertainty-weighted triangulation.**  The anisotropic covariance is simplified to a per-view log-variance, making the final triangulation:
   ```
   argmin_X Σ_v m_v · exp(-λ_v) · ||π_v(X) − x_v||²
   ```
4. **Warm-start from the best checkpoint.**  The strong 9.32 mm PP model is used to initialize the encoder; new heads start from scratch and the encoder is frozen for a short burn-in phase to preserve the geometric prior.

### 3.3 Why this is the right next step

- **Accuracy headroom:** Bayesian tri v2 already reaches the threshold; unification should push clean MPJPE toward **≤ 8.0 mm** while keeping parameter count under ~1.2 M.
- **Story clarity:** A single architecture with named, well-defined submodules is easier to explain and diagram than a sequence of incremental variants.
- **Robustness:** Visibility gating + uncertainty weighting + graph attention together address the occlusion failure mode that currently degrades the model most (view dropout → 18.15 mm).
- **Deployability:** The module still exposes the same `MultiViewFusionPlugin` interface; no downstream code changes.

---

## 4. Revised method section for the paper

### 4.1 Input and notation

- `x_vj ∈ R²` — 2D keypoint for view `v` and joint `j`.
- `c_vj ∈ [0,1]` — detector confidence.
- `K_v, R_v, t_v` — intrinsic and extrinsic calibration.
- Output: 3D pose `X_j ∈ R³`, per-view visibility `m_vj`, per-view uncertainty `λ_vj`, and confidence `q_j`.

### 4.2 Intrinsic self-calibration head

```
(Δcx, Δcy, Δf) = IntrinsicMLP( pool_t(f_v) )
K'_v = K_v corrected by (Δcx, Δcy, Δf)
```

Corrections are bounded (`tanh`) and supervised against the inverse of the training perturbation.  This handles the dominant field failure mode: principal-point drift.

### 4.3 Visibility head

```
m_vj = sigmoid( MLP_vis( [x_vj, c_vj, ray_vj, cam_embed_v] ) )
```

`m_vj` is used as an attention mask in view-attention and as a multiplicative weight in triangulation.  A `min_visible_views` guard prevents degenerate triangulation.

### 4.4 Factorised (T × V × J) attention block

After ray-aware feature extraction, tokens form a 3-D grid `(T, V, J)`.  We apply separate Transformer layers along each axis:

- **Temporal layers** — smoothness and motion context.
- **View layers** — multi-view consistency and occlusion reasoning (masked by `m_vj`).
- **Joint layers** — skeleton graph message passing.

This factorisation reduces complexity from `O((TVJ)²)` to `O(T² + V² + J²)` per axis.

### 4.5 Uncertainty-weighted triangulation

```
w_vj = c_vj · m_vj · exp(-λ_vj)
X_raw = weighted_DLT( {x_vj, w_vj, K'_v, R_v, t_v} )
X_gn = GaussNewton(X_raw; λ, m; 1–2 steps)
```

### 4.6 Residual refinement

```
X = X_gn + MLP( [pool(factorised_features), X_gn] )
```

The residual head now receives factorised spatio-temporal-joint context, making the correction smaller and more structured.

### 4.7 Training objective

```
L = L_3D_MSE
  + λ_vis L_visibility_BCE
  + λ_unc L_uncertainty_NLL
  + λ_pp L_intrinsic_correction
  + λ_epi L_epipolar
  + λ_bone L_bone_length
  + λ_vel L_velocity_smoothness
```

Camera perturbation is retained and slightly widened (rotation ±0.7°, translation ±7 mm, focal ±1.5%, principal point ±6 px) to match the combined robustness targets.

---

## 5. Expected experimental narrative

### 5.1 MPI-INF-3DHP targets

| Metric | Bayesian tri v2 best | OmniMultiViewFusion target | Comment |
|--------|---------------------|----------------------------|---------|
| Clean MPJPE | 8.35 mm (ensemble) | **≤ 8.0 mm** single model | Unification should beat the ensemble without ensembling. |
| PA-MPJPE | 5.29 mm | **≤ 5.0 mm** | Graph attention should improve pose alignment. |
| View dropout 30% | 18.15 mm | **≤ 13 mm** | Visibility gating + uncertainty. |
| Joint occlusion 20% | 14.56 mm | **≤ 12 mm** | Skeleton graph propagation. |
| rot_0.5° | 16.89 mm | **≤ 14 mm** | Stronger intrinsic head + perturbation. |
| Runtime | 12–195 clips/s | ≥ 100 clips/s | Factorised attention is faster. |

### 5.2 Human3.6M cross-dataset

| Metric | Current | Target |
|--------|---------|--------|
| Clean MPJPE | 5.24 mm | **≤ 4.5 mm** |
| PA-MPJPE | 4.84 mm | **≤ 4.0 mm** |
| Cross-dataset H36M→MPI | 101 mm | **≤ 30 mm** (long-term) |

### 5.3 Variable-view curve

Target `MPJPE@k` for `k = 2..14` views on MPI-INF-3DHP:
- `k=2`: ≤ 25 mm
- `k=4`: ≤ 15 mm
- `k=8`: ≤ 10 mm
- `k=14`: ≤ 8 mm

This gives a strong figure for the paper: a single model gracefully degrades as views are dropped.

---

## 6. Updated contribution list

1. **Unified OmniMultiViewFusion module.** A single architecture combining intrinsic self-calibration, visibility gating, factorised (T×V×J) attention, uncertainty-weighted triangulation, and residual refinement.
2. **Graph-constrained joint attention.** Skeleton-aware message passing replaces dense joint self-attention, improving anatomical plausibility under occlusion.
3. **Uncertainty-aware triangulation.** Per-view log-variance weights make the DLT step robust to noisy or partially occluded views without hard thresholds.
4. **Warm-start recipe.** A practical strategy to unify previously isolated modules without destabilising a strong 9.32 mm baseline.
5. **Stronger MotionFlow plugin.** Same `MultiViewFusionPlugin` interface, with richer uncertainty and provenance metadata.

---

## 7. Paper arc update

### New / updated sections

**Abstract (updated).**  "We present OmniMultiViewFusion, a unified multi-view 3D human pose estimator that predicts per-view intrinsic corrections, visibility, and uncertainty, then triangulates with a factorised spatio-temporal-joint Transformer.  On MPI-INF-3DHP it reaches **≤ 8.0 mm** MPJPE while improving robustness to occlusion, view dropout, and calibration drift."

**Method — new subsection 4.8.**  "OmniMultiViewFusion: unifying calibration, visibility, and factorised attention."

**Experiments — new subsections.**
- Variable-view MPJPE@k curve.
- Extended robustness matrix (occlusion, dropout, calibration, noise, outliers).
- Cross-dataset zero-shot H36M→MPI.

**Figures — new.**
- OmniMultiViewFusion architecture diagram (2D keypoints → intrinsic correction → visibility → factorised T×V×J attention → uncertainty → weighted triangulation → GN → residual).
- Variable-view MPJPE@k curve.
- Robustness heatmap comparing Bayesian tri v2, PP-corrected baseline, and OmniMultiViewFusion.

---

## 8. Risk register update

| Risk | Status | Mitigation |
|------|--------|------------|
| Unified model fails to improve clean accuracy | Open | Warm-start from 9.32 mm checkpoint; freeze encoder for 5 epochs. |
| Factorised attention is still too slow | Open | Profile each axis; fall back to temporal + view only. |
| Visibility head collapses to all-ones | Open | Auxiliary BCE loss + curriculum on view-dropout rate. |
| Graph attention incompatible with variable views | Open | Rebuild edge index for active joint subset. |
| Graph attention adds too much compute | Open | Use 1 graph layer only; benchmark on RTX 4090. |
| Cross-dataset gap remains | Open | Add domain-adaptation wrapper after fusion module. |

---

## 9. Open questions for iter18 and beyond

1. Should the uncertainty head be anisotropic (full 2×2 covariance, as in Bayesian tri v2) or isotropic log-variance?  Isotropic is cheaper; anisotropic keeps the epipolar-auxiliary-loss story.
2. Should the graph-joint attention be applied before, after, or in parallel to view/temporal attention?  Order affects gradient flow.
3. How large should the visibility-head auxiliary loss weight be?  Too high may suppress useful views; too low lets the head stay lazy.
4. Is the warm-start checkpoint (`ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth`) the right starting point, or should we start from the 8.35 mm ensemble member?
5. Does OmniMultiViewFusion need a separate cross-dataset domain-adaptation head, or can the same factorised backbone generalise if trained on H36M + MPI jointly?

---

## 10. Related files and prototypes

- Design sketch: `docs/design_omniview_fusion.md`
- Paper story (parent): `docs/icra_cvpr_2027_paper_story.md`
- Prototype stub: `experiments/prototypes/swarm_iter18/omnimultiview_fusion_smoke.py`
- Training script (to be written): `experiments/train_omnimultiview_mpiinf3dhp.py`
- Evaluation: `experiments/prototypes/run_extended_robustness_matrix.py`

---

## 11. Next concrete steps

1. Implement `OmniMultiViewFusion` model in `motionflow_mv/fusion/` or as an isolated prototype first.
2. Run a smoke run (d=48, 10 epochs, MPI-INF-3DHP) to confirm no NaNs and that clean MPJPE stays within 5% of the 9.32 mm baseline.
3. Run a full 20–30 epoch run (d=128) and report clean / PA / robustness metrics.
4. Generate variable-view MPJPE@k curve.
5. Update `docs/icra_cvpr_2027_paper_story.md` with final OmniMultiViewFusion numbers and fold this P11 section into the main paper story.

# v27 / Next Iteration Decision Matrix

This document records the candidate next steps after v25 (geometry fusion) and
v26 (temporal geometry fusion). The actual path is gated by the first-epoch
`val_MPJPE` of v25 small and the v26 small comparison.

## 1. Decision gates

| Gate | Condition | Likely action |
|------|-----------|---------------|
| G1 | v25 small `val_MPJPE` < v18 baseline (20.24 mm) | v25 geometry fusion is promising; keep full/ablation running and launch v26 |
| G2 | v25 small `val_MPJPE` ≈ v18 baseline (within ~1 mm) | Geometry fusion adds no clear benefit; debug or simplify v25 before v26 |
| G3 | v25 small `val_MPJPE` > v18 baseline (>21 mm) | Stop v25 full/ablation; redesign geometry fusion module |
| G4 | v26 small `val_MPJPE` < v25 small `val_MPJPE` | Temporal extension helps; invest in v27 temporal variants |
| G5 | v26 small `val_MPJPE` ≥ v25 small `val_MPJPE` | Temporal window does not help; focus on other axes |

## 2. Candidate v27 directions

### 2.1 Uncertainty-aware depth proposals (high priority)
- **Motivation:** v25 learns a fixed number of depth proposals per ray. A learned
  per-ray/per-joint uncertainty can weight proposals adaptively.
- **Change:** Replace the fixed `n_ray_samples` depth grid with a Gaussian mixture
  or learned per-ray depth distribution.
- **Cost:** Small (one extra head); warm-startable (identity when weights zero).
- **Risk:** May overfit on small WebBridge subsets.

### 2.2 Camera refinement inside the fusion loop (medium priority)
- **Motivation:** `CameraRefinementV26` already exists but is not necessarily
  wired into the main training path. Jointly refining cameras and pose inside the
  fusion block could improve robustness to calibration drift.
- **Change:** Add `CameraRefinementV26` as an optional block between geometry
  attention and depth triangulation, gated by a flag and a learnable mixing weight.
- **Cost:** Moderate; needs careful gradient flow through `K, R, t`.
- **Risk:** Camera refinement can collapse to trivial solutions if not constrained.

### 2.3 Diffusion-based pose refiner on top of v26 (medium priority)
- **Motivation:** `DiffusionPoseRefiner` (v20) exists but is under-tested with the
  geometry-first pipeline. It could clean up the residual error after geometry
  fusion.
- **Change:** Add the refiner after `TemporalGeometryFusionV26`, conditioned on
  the geometry features.
- **Cost:** High (training time, inference cost).
- **Risk:** May be overkill if v25/v26 already reach the target MPJPE.

### 2.4 Variable-view curriculum + outlier hardening (low priority)
- **Motivation:** Variable-view training and outlier-view augmentation are already
  in v25. A more aggressive curriculum could improve few-view performance.
- **Change:** Ramp the number of views from 2 to 14 over more epochs; add hard
  negative outlier views.
- **Cost:** Low (training schedule change only).
- **Risk:** Longer training, marginal gain.

## 3. Recommended order

1. **If G1 and G4 hold:** start with **2.1 uncertainty-aware depth proposals**
   (v27a) because it is the smallest, warm-startable extension that directly
   improves the geometry core.
2. **If G1 but not G4:** start with **2.2 camera refinement** or **2.4 variable-view
   curriculum** to improve per-frame robustness before adding temporal cost.
3. **If G2 or G3:** stop v25 full/ablation and redesign the geometry attention
   (e.g., ablate epipolar vs. ray-intersection bias, reduce depth-proposal count,
   fix gradient issues) before defining v27.

## 4. Success criteria for v27

- Reduce `val_MPJPE` by at least **10%** over the best v25/v26 baseline.
- Maintain or improve variable-view performance at 2, 4, 8, and 14 views.
- Add <10% model parameters and <20% training time compared to v25.

## 5. Immediate next steps

- [ ] Wait for v25 small first-epoch `val_MPJPE`.
- [ ] Launch v26 small once a GPU frees.
- [ ] Compare v25/v26 small results and select a direction from §2.

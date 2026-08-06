# Occlusion-Robust Multi-View Fusion

## 1. Survey

In calibrated multi-view human pose estimation, occlusions, out-of-frame joints, and detector failures produce corrupted 2D observations. Naive triangulation that treats all views equally therefore drifts toward the bad observations. Occlusion-robust fusion estimates *per-view visibility* or *uncertainty*, then down-weights or masks unreliable views before or during triangulation.

The current MotionFlow-MV pipeline in `motionflow_mv/fusion/ray_attention_model.py` already has the basic machinery: `RayAttentionFusionModel` predicts per-view per-joint weights `w ∈ (0,1)`, multiplies them by input confidences, and feeds the result to a differentiable weighted DLT triangulator. Synthetic experiments (`docs/design_v3.md`) show this design tolerates 1–2 occluded views (MPJPE grows from 3.6 mm to 5.7 mm). However, on real Shelf/Campus data the model only sees GT-derived 2D points and raw visibility scores loaded by `motionflow_mv/data/shelf_loader.py`; it does not yet learn an explicit *occlusion mask* or an *uncertainty model* beyond the confidence multiplier.

Prior work has three strands: (i) geometric robust estimators (RANSAC, M-estimator triangulation, epipolar consistency), (ii) learned per-view weights or masks from attention/CNN/MLP heads, and (iii) probabilistic fusion treating 2D detections as Gaussian measurements. Strong practical systems combine a geometric inductive bias with a learned visibility/uncertainty head. Our ray-aware attention architecture is well positioned: the DLT layer supplies geometry, while the attention head can learn to recognize corrupted views.

## 2. Actionable Recommendations

1. **Add an explicit learned occlusion/uncertainty head.** The model currently outputs one scalar per `(view, joint)` and multiplies it by the input confidence. Split it into a *visibility* mask `v_vj ∈ [0,1]` (soft binary, trained with auxiliary BCE against GT or proxy occlusion labels) and a *precision* scalar `ρ_vj > 0`. Use the product `v_vj · ρ_vj · c_vj` as the DLT weight. This separates “is the joint visible?” from “is the 2D measurement precise?”, which matters when detectors are confident but the joint is occluded.

2. **Use GT-derived occlusion labels as supervision during real training.** `shelf_loader.py` already loads `visibility_list` and broadcasts it as confidences. Extend the loader to expose a boolean or continuous occlusion mask `o_vj` per view (e.g., threshold the visibility score). Add an small auxiliary loss `L_occ = BCE(v_hat_vj, o_vj)` weighted by ~0.1 so the attention head learns to predict visibility, not only 3D accuracy. This is the fastest path to real-data occlusion awareness.

3. **Replace the confidence product with uncertainty-based weighting.** Weighted DLT currently assumes weights are inverse variances only informally. Add a small MLP branch that outputs log-variance `log σ²_vj` and compute weights as `w_vj = exp(-log σ²_vj) · v_vj`. This turns the triangulator into a heteroscedastic maximum-likelihood estimator and gives calibrated uncertainties for downstream fusion (temporal or residual refiner). It also yields interpretable per-joint uncertainty for the paper.

4. **Augment training with realistic occlusion patterns.** The synthetic generator already injects 10% random occlusion. On real Shelf/Campus, add view-dropping augmentation: randomly zero out 0–2 views per joint and mask their 2D coordinates and confidences. This prevents the learned weights from overfitting to the fixed 4-camera rig. Keep the DLT loss only over non-masked views.

5. **Add epipolar consistency as a visibility prior before the learned head.** Compute a per-view residual relative to an epipolar line from a reference view, and feed it as an extra feature (epipolar distance, reprojection error) to the weight head. This gives a geometric cue for occlusion independent of raw confidence, which helps when detector confidence is miscalibrated.

## 3. Potential Risks

- **Overfitting to synthetic occlusion.** The synthetic generator drops joints uniformly across views. Real occlusions are view-dependent and correlated with pose and scene layout. Without real GT occlusion labels, the learned mask may not transfer to Shelf/Campus.
- **DLT degeneracy when too many views are masked.** If the head masks all views for a joint, the weighted DLT system becomes rank-deficient. Add a `mask_max_fraction` limit or an unweighted DLT fallback for effective views < 2.
- **Scale/unit drift.** Shelf is in millimeters and synthetic data in meters. `train_ray_attention_real.py` scales by 0.01 and the model uses camera tensors; if uncertainty or ray embeddings are not unit-normalized, occlusion-aware weights may become dataset-dependent. Maintain the per-plugin `input_scale` / `output_scale` contract from `design_v3.md`.

## 4. Fit to the Paper Plan

This work directly addresses the CVPR/ICRA goal of “geometry-aware learned fusion that beats DLT.” A strong occlusion-robust fusion result supports two claims: (1) the attention plugin is more accurate than geometry-only baselines and more robust to missing views; and (2) per-joint uncertainty/visibility estimates can be exported to downstream modules as principled weights. Recommended paper experiments: (a) report MPJPE on Shelf/Campus with 0/1/2 random dropped views; (b) compare uncertainty-weighted DLT against plain confidence-weighted DLT and the current `ray_attention` baseline; (c) visualize learned visibility masks for interpretability.

# Ablation Study Design: Views, Joints, Noise, and Model Size

## Survey

The MotionFlow multi-view extension introduces a **ray-aware attention fusion** module (`ray_attention`) that predicts per-view, per-joint weights and feeds them into a differentiable weighted DLT triangulator. Initial validation on synthetic data with 0.5 px Gaussian noise reports `ray_attention` MPJPE of 0.0021 m versus 3.6807 m for the older flattened-projection-matrix `attention` model. The `RayAttentionFusionModel` exposes the key variables `j`, `d`, `n_views`, and `n_heads`, making it easy to ablate the factors that most affect accuracy, robustness, and deployment cost.

This report proposes a focused ablation protocol around four axes: **number of views**, **joint set granularity**, **2D keypoint noise level**, and **model size**. The goal is to identify when the learned attention weights improve over a strong geometric baseline (confidence-weighted DLT) and when they are unnecessary.

## Actionable Recommendations

### 1. Number of Views: 2→8 Sweep with View-Dropout

Use `experiments/generate_synthetic_multiview_dataset.py` to create rigs with `V = 2, 3, 4, 6, 8` views. Train a single `d=64` `ray_attention` checkpoint on `V=8` data, then evaluate with random view subsets at test time. Plot MPJPE and per-view weight entropy against `V`.

*Expected outcome*: a knee around `V=4`; graceful degradation below it and higher entropy for informative baseline views. Once Shelf/Campus data is available, mirror the sweep on real baselines (`V=5` for Shelf, `V=3/4` for Campus), using synthetic augmentation to fill missing cardinalities.

### 2. Joint Granularity: Reduced, Full, and Over-Complete Skeletons

Compare:

- a reduced skeleton (8–10 key joints),
- the full skeleton used by `shelf_loader.py` (17 joints),
- an over-complete skeleton (e.g., 25 joints from a standard SMPL regressor).

Report MPJPE and per-joint weight reliability. The hypothesis is that peripheral joints (wrists, ankles) benefit most from attention, while torso joints are already well handled by simple DLT.

### 3. Noise Level: Structured Corruption Schedule

Inject Gaussian 2D noise at `σ = 0, 0.5, 1.0, 2.0, 5.0` px, plus 10% occlusion and 2% random outliers (matching `eval_ray_attention_robustness.py`). Evaluate `ray_attention`, confidence-weighted DLT, and the old `attention` baseline. Target questions: at what noise level does learned weighting outperform the geometric baseline, and where does it break? Keep noise units in pixels to match the real loader.

### 4. Model Size: Embedding Dimension and Head Count Grid

Vary `d ∈ {32, 64, 128, 256}` and `n_heads ∈ {2, 4, 8}` (with `n_heads` dividing `d`). Fix `V=4` and `σ=1.0` px. Report parameter count, approximate FLOPs, training time per epoch, and validation MPJPE. The objective is the smallest model whose performance plateaus; the current default `d=64, n_heads=4` should be justified empirically, not by convention.

### 5. Cross-Configuration Generalization

Train on one noise profile and test on another (e.g., train at `σ=1.0` px, test at `σ=5.0` px). Also train on `V=8` and evaluate at `V=4`. This checks whether the attention head learns a general weighting strategy or merely memorizes the training distribution. If cross-configuration performance drops, add noise/view dropout as augmentation.

## Potential Risks

- **Synthetic-to-real gap**: controlled ablations will first be synthetic; numbers may not transfer directly to Shelf/Campus because the synthetic generator randomizes rigs rather than copying real baselines.
- **Fixed real baselines**: Shelf/Campus have fixed camera counts, so the view-count ablation cannot exceed the physical setup without synthetic augmentation.
- **Interaction effects**: noise, view count, and model size are coupled. A small model may suffice for clean data but fail under high noise. The ablation matrix must be wide enough to expose interactions without exhausting compute.
- **Strong DLT baseline**: confidence-weighted DLT is already very accurate on clean data. The ablation must therefore emphasize occluded views, noisy keypoints, and extreme joint angles—regimes where learned weighting has the most to offer.

## Fit into the Paper Plan

These ablations belong in the **experimental evaluation** section. They support the central claim that ray-aware attention plus differentiable DLT provides a calibrated, robust multi-view fusion layer:

- **View-count ablation** shows scalability and justifies the real setup.
- **Joint ablation** justifies the chosen skeleton resolution.
- **Noise ablation** demonstrates practical robustness for markerless pose estimation.
- **Model-size ablation** supports deployability claims.

Together, the results provide the quantitative backbone for the ICRA/CVPR 2027 submission and reduce reviewer pushback that the design choices are ad-hoc.

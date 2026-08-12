# v33 Design Proposal: View Synthesis / Neural Rendering for Data Augmentation

**Slug:** `view_synthesis_augmentation`  
**Title:** View synthesis / neural rendering for data augmentation  
**Date:** 2026-08-08  
**Target conference:** ICRA/CVPR 2027  
**Status:** Design proposal — no source files modified.

---

## 1. Problem Statement and Motivation

The MotionFlow-MultiView v5/v31/v32 pipeline is trained almost exclusively on real multi-view captures (H36M with 4 cameras, MPI-INF-3DHP with 14 cameras). While the model already supports variable-view training via random view subsets and permutations, the *distribution of camera viewpoints* is still tightly coupled to the fixed rig geometries of the training datasets. This limits:

1. **Generalization to unseen camera rigs** — a deployment camera setup may have very different baselines, heights, or azimuth distributions than H36M/MPI.
2. **Variable-view robustness** — sampling subsets of the same 14-view rig does not teach the model how to fuse from rigs with different geometry.
3. **Data scarcity** — expanding to new subjects or actions requires collecting or licensing additional multi-view data.

**Idea:** given a 3D pose sequence and a calibrated camera rig, use lightweight *view synthesis / neural rendering* to synthesize plausible 2D keypoint observations from *virtual camera viewpoints* that never existed in the original capture. The synthesized samples are fed into the same `(x, y, K, R, t)` training pipeline as real data, increasing viewpoint diversity without capturing new footage.

This direction is distinct from the existing Gaussian-splatting retry (a regularizer on predicted 3D uncertainty) and the synthetic SMPL/AMASS generator (procedural body motions). v33 focuses on **novel-view synthesis from real 3D pose ground truth** as a data-augmentation strategy.

---

## 2. Proposed Architecture Changes

### 2.1 New module: `motionflow_mv/data/view_synthesis_augmentation_v33.py`

A self-contained augmentation module that can be toggled on/off and is invoked inside the training script before the forward pass.

**Public classes / functions**

| Name | Purpose |
|------|---------|
| `VirtualCameraSampler` | Sample virtual camera extrinsics (azimuth, elevation, radius, roll) given a target distribution (H36M-like, MPI-like, or uniform). |
| `NeuralViewRendererV33` | Lightweight MLP that maps `(joint_3d, camera_position, view_dir)` → `(uv_offset, confidence)` to render more realistic 2D keypoints than pure geometric projection. |
| `GeometryViewRendererV33` | Deterministic pinhole projection of 3D joints with optional 2D noise, occlusion, and confidence decay. |
| `ViewSynthesisAugmenter` | Orchestrates the above: decides when to synthesize, how many virtual views, and merges them with real views. |
| `blend_virtual_views` | Helper to merge real `(B, T, V, J, 3)` clips with virtual `(B, T, V', J, 3)` clips into a single padded tensor and a corresponding `view_mask`. |

**Key design decisions**

- The renderer operates on **2D keypoints + confidences**, not raw images, so it stays lightweight and avoids image-based NeRF/GS training costs.
- It reuses the existing canonical `(x, y, K, R, t)` tuple format used by `TemporalClipDataset` and `WebBridgeMixedDataset`.
- A **geometry-only fallback** is provided so the model can be trained even before a neural texture is ready.

### 2.2 Integration into `OmniMultiViewFusionV5`

No changes to the model class are required for the first milestone. The augmentation is applied in the training script (`experiments/train_omniview_fusion_v5_webbridge_multi.py`) inside the `compute_loss` closure, immediately after the existing `augment_clip` call.

Pseudo-flow:

```
real_x, view_mask = augment_clip(real_x, ...)

if args.use_view_synthesis_aug_v33 and random.random() < args.v33_aug_prob:
    virtual_x, virtual_K, virtual_R, virtual_t = view_synthesis_augmentation_v33(
        y, K, R, t,
        n_virtual_views=args.v33_n_virtual_views,
        mode=args.v33_virtual_view_mode,
    )
    x = concatenate(real_x, virtual_x, dim=2)  # along view dimension
    K = concatenate(K, virtual_K, dim=1)
    R = concatenate(R, virtual_R, dim=1)
    t = concatenate(t, virtual_t, dim=1)
    view_mask = extend_view_mask(view_mask, n_virtual_views)

out = model(x, K=K, R=R, t=t, view_mask=view_mask)
```

For a future milestone, an optional **neural rendering consistency loss** could be added inside the model:

- `motionflow_mv/losses/view_synthesis_loss_v33.py`
  - `neural_render_consistency_loss(pred_3d, rendered_2d, K, R, t)` — encourages the model's 3D prediction to reproject consistently with the rendered virtual views.

This is **not** required for the first smoke test.

### 2.3 New CLI flags in `train_omniview_fusion_v5_webbridge_multi.py`

```text
--use_view_synthesis_aug_v33          Enable view-synthesis augmentation (default: False)
--v33_aug_prob                         Per-batch probability of applying augmentation (default: 0.5)
--v33_n_virtual_views                  Number of virtual views to synthesize (default: 2)
--v33_virtual_view_mode                {geometry, neural, hybrid} (default: geometry)
--v33_camera_radius_range              Min/max camera distance in meters (default: "2.0,6.0")
--v33_camera_height_range              Min/max camera height in meters (default: "0.5,2.5")
--v33_use_neural_texture               Learn a small per-joint neural texture (default: False)
--v33_neural_texture_hidden            Hidden dim for the neural texture MLP (default: 64)
--v33_geometry_noise_std               2D noise on geometrically projected points (default: 1.0 px)
--v33_occlusion_rate                   Probability of occluding a virtual (view, joint) pair (default: 0.1)
--v33_loss_weight                      Weight for optional neural-render consistency loss (default: 0.0)
```

These flags follow the existing convention of version-prefixed toggles (`v25_*`, `v29_*`, `v32_*`).

### 2.4 Data / preprocessing requirements

- The augmentation requires **3D ground-truth poses** (`y`) and calibrated cameras (`K, R, t`) for every training clip. This is already available in the WebBridge `.npz` files used by `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`.
- Virtual camera intrinsics are sampled to roughly match the focal lengths and principal points of the real rig.
- For the mixed loader, virtual views are added **after** the H36M/MPI domain padding, so the domain-aware view masking logic remains valid.

---

## 3. Training Command / Ablation Flags

### 3.1 Smoke test (CPU / RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_view_synthesis_aug_v33 \
    --v33_aug_prob 0.5 \
    --v33_n_virtual_views 2 \
    --v33_virtual_view_mode geometry
```

### 3.2 Full v33 ablation on WebBridge mixed data

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --domain_aware_view_curriculum \
    --use_view_synthesis_aug_v33 \
    --v33_aug_prob 0.5 \
    --v33_n_virtual_views 2 \
    --v33_virtual_view_mode geometry \
    --v33_camera_radius_range "2.0,6.0" \
    --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 \
    --output outputs/omniview_fusion_v33_view_synthesis_aug.pth
```

### 3.3 Ablation matrix

| Run | Mode | Virtual views | Notes |
|-----|------|---------------|-------|
| `v33_geometry_2v` | `geometry` | 2 | Baseline augmentation; pure pinhole projection. |
| `v33_geometry_4v` | `geometry` | 4 | More virtual views; tests diminishing returns. |
| `v33_neural_2v` | `neural` | 2 | Adds learnable per-joint neural texture. |
| `v33_hybrid_2v` | `hybrid` | 2 | Geometry + neural texture; likely best realism/cost trade-off. |
| `v32_baseline` | — | 0 | Existing v32 combined run for comparison. |

---

## 4. Expected Metrics and Baseline to Beat

### 4.1 Baseline

The most recent v32 baseline metrics from `docs/swarm_iter_next/20_agent_direction_review.md`:

- **MPI-INF-3DHP clean:** MPJPE = **9.32 mm**, PA-MPJPE = **5.37 mm**
- **Variable-view curve:** currently running; target is a monotonic decrease from k=2 to k=14.

The v32 A800 queue (`scripts/launch_v32_a800_queue.py`) is training:

- `v32_domain_aware_view_curriculum`
- `v32_trajectory_consistency_refiner`
- `v32_combined`
- `v32_ray_attention`
- `v32_physical_alignment`

We will use the best v32 checkpoint as the baseline.

### 4.2 Expected v33 improvements

| Metric | Baseline (v32) | v33 target | How it helps |
|--------|----------------|------------|--------------|
| MPI-INF-3DHP clean MPJPE | 9.32 mm | ≤ 9.0 mm | More viewpoint diversity reduces overfitting to fixed rig. |
| H36M clean MPJPE | ~11–13 mm | ≤ 10.5 mm | Virtual views break H36M's 4-view symmetry. |
| Variable-view k=2 | TBD | No regression | Virtual multi-view training still enforces few-view fusion. |
| Variable-view k=14 | TBD | Monotonic ↓ | Extra virtual views act as regularization. |
| Camera perturbation robustness | TBD | ≥ 5% relative improvement | Virtual camera sampling explicitly trains calibration-robust features. |

### 4.3 Minimum success criteria

1. **No regression:** v33 with `v33_aug_prob=0` matches the v32 baseline within 0.2 mm.
2. **Clean gain:** at least one v33 variant improves MPI-INF-3DHP clean MPJPE by ≥ 0.3 mm over v32.
3. **Robustness gain:** variable-view curve remains monotonic and camera-perturbation MPJPE improves by ≥ 5%.

---

## 5. Risks / Unknowns

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Geometry-only rendering is too trivial** and adds no information beyond existing augmentation. | Medium | Low | Keep the geometry path cheap; switch to `neural`/`hybrid` if smoke tests show no gain. |
| **Neural texture training is unstable or slow** and dominates GPU budget. | Medium | Medium | Default to `geometry` mode; neural path is gated by `--v33_use_neural_texture`. |
| **Virtual views introduce unrealistic 2D joint distributions** (e.g., extreme foreshortening) that hurt rather than help. | Medium | Medium | Clamp azimuth/elevation/radius ranges; validate reprojection errors on a held-out real set. |
| **Mixed-loader domain padding breaks** when concatenating real and virtual views. | Low | High | Extend the domain mask and `view_mask` carefully; add a dedicated smoke test. |
| **Diminishing returns** — adding 2–4 virtual views helps, but 4+ adds noise. | Medium | Low | Ablation matrix starts at 2 views. |
| **Conflicts with existing augmentation** (outlier injection, occlusion, variable-view) that duplicate or cancel effects. | Low | Medium | Run controlled ablations: v33-only, existing-only, combined. |

### 5.1 Open questions

1. Should virtual views be sampled per-clip or per-epoch? Per-clip gives more diversity; per-epoch is cheaper.
2. Should the neural texture be shared across all joints, or should each joint have its own appearance embedding?
3. How do we best evaluate **novel-view synthesis quality** independently of pose accuracy? A small held-out set of real camera views could be used for a reprojection-MSE diagnostic.
4. Can we leverage the existing `gaussian_splatting_pose_loss.py` to regularize the rendered virtual views, or is that redundant with the v25 geometry fusion?

---

## 6. Implementation Roadmap (First 2 Weeks)

1. **Week 1 — Geometry-only smoke:**
   - Implement `ViewSynthesisAugmenter` with `geometry` mode.
   - Add CLI flags to `train_omniview_fusion_v5_webbridge_multi.py`.
   - Run CPU smoke and RTX 4090 smoke on `--v33_virtual_view_mode geometry`.

2. **Week 1–2 — Neural renderer prototype:**
   - Implement `NeuralViewRendererV33` as a small MLP with per-joint embeddings.
   - Add optional `view_synthesis_loss_v33.py` consistency loss.

3. **Week 2 — Ablation & benchmark:**
   - Run `v33_geometry_2v`, `v33_geometry_4v`, `v33_neural_2v`, `v33_hybrid_2v` on RTX 4090 smoke/full.
   - Compare against the v32 baseline on clean + variable-view + camera-perturbation protocols.
   - Update this proposal with measured numbers.

---

## 7. Related Files

- Training script: `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- Model: `motionflow_mv/fusion/omniview_fusion_v5.py`
- Mixed-data manifest: `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`
- A800 launch script: `scripts/launch_v32_a800_queue.py`
- Prior synthetic-data work: `docs/swarm_iter_next/design_synthetic_amass_augmentation/report.md`
- Prior Gaussian-splatting regularizer: `docs/swarm_iter_next/iter_gaussian_splatting_retry_plan.md`

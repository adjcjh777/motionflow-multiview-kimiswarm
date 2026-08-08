# v33 proposal: `camera_geometry_embedding` — Camera geometry embedding and calibration-aware fusion

## Problem statement and motivation

`OmniMultiViewFusionV5` already has two camera-aware components: a camera-conditioned view embedding (`CameraConditionedViewEmbedding` / v31 upgrade) and a geometry-biased hierarchical encoder (`HierarchicalViewEncoderV31`).  However, the camera signal is still used *additively and late*:

* The embedding only summarises each camera (or pairwise camera geometry) as a single `d`-D vector that is added once to every joint token.  It does not exploit the **per-joint ray geometry** (camera centre + ray direction for each 2-D observation) that is the natural common language between cameras and 3-D joints.
* The geometry bias in `HierarchicalViewEncoderV31` is handcrafted from epipolar distance and ray-intersection quality, but it is not **learned end-to-end** from the camera parameters themselves; the model has no explicit “calibration-aware fusion” path that lets it, for example, down-weight a view whose predicted calibration is inconsistent with the others.
* Variable-view training and calibration perturbations are becoming central to the WebBridge/H36M/MPI mixed-dataset setup, yet the camera embedding currently ignores the `view_mask` (the v31 embedding accepts it but it is a no-op) and there is no loss that teaches the model to be robust to *incorrect* calibration.

Consequently, when views are dropped, permuted, or perturbed, the model must rediscover geometric relationships from scratch in the transformer.  A richer, per-joint **camera geometry embedding** fused at multiple stages should improve generalisation, especially for variable-view inference and calibration-robust evaluation.

## Proposed architecture changes

### 1. `CameraGeometryEmbeddingV33` — per-joint ray-aware camera embedding

**New file:** `motionflow_mv/fusion/camera_geometry_embedding_v33.py`

**Class:** `CameraGeometryEmbeddingV33(nn.Module)`

The module replaces/extends the v31 camera embedding.  It receives `K, R, t` *and* the 2-D observations `points_2d` and `confidences`, and returns:

* `view_embedding`: `(B, V, d)` — same shape as the current v31 embedding, but richer.
* `joint_geometry_feat`: `(B, T, V, J, d)` — per-joint, per-view geometry tokens that can be added to the encoder tokens.

**Inputs encoded:**

| Signal | Why it matters |
|---|---|
| Normalised intrinsics (`fx, fy, cx, cy`) | Camera-specific scale and optical centre. |
| Camera centre `C = -R^T t` and optical axis | Same as v31 local branch, but used again below. |
| Pairwise baseline / relative rotation / optical-axis cosine | Same as v31 pairwise branch. |
| **Per-joint ray direction** `d = R^T K^{-1} [u, v, 1]^T` | The actual geometric cue that links a 2-D keypoint to a 3-D point. |
| **Per-joint ray confidence** | Allows the embedding to suppress occluded/uncertain rays. |
| **Epipolar/ray-interaction summary** | Distance to scene centroid along ray, ray pair shortest distance, and angle to other views, summarised into a small vector per (view, joint). |

**Design details:**

* Keep the v31 two-branch structure (local camera descriptor + pairwise view-geometry self-attention) for the `view_embedding` output, but add a third branch that builds per-joint ray features.
* The third branch is a small MLP operating on `direction, centre, confidence, C_rel, dist_to_centroid` and produces `(B, T, V, J, ray_hidden)`.
* A final linear projection to `d` is zero-initialised, so the new branch is a no-op at init and warm-start from v31 checkpoints is possible.
* Respect `view_mask` in the pairwise self-attention (the v31 implementation already accepts the argument but does not use it for the local branch; v33 fixes this).

### 2. `CalibrationAwareFusionV33` — geometry-aware cross-view fusion

**New file:** `motionflow_mv/fusion/calibration_aware_fusion_v33.py`

**Class:** `CalibrationAwareFusionV33(nn.Module)`

This module sits **between** the ray tokenizer / geometry-aware cross-view attention and the final triangulation in `OmniMultiViewFusionV5`.  It consumes the per-joint geometry tokens from `CameraGeometryEmbeddingV33` and refines both:

1. **Cross-view attention weights** inside the hierarchical encoder, by adding a learned residual geometry bias:
   ```
   bias_v33 = MLP(concat[content_tokens, joint_geometry_feat])
   attention_scores = content_scores + sigmoid(gate) * bias_v33
   ```
   The gate is initialised to near-zero so the block is identity at init.

2. **Triangulation weights**, by predicting a per-(view, joint) calibration reliability scalar `gamma` that multiplies the existing triangulation weights:
   ```
   w'_v = softmax(log w_v + log gamma_v)
   ```
   This lets the model explicitly down-weight views whose geometry is inconsistent with the current estimate, going beyond the current outlier detector.

**Optional calibration-consistency loss:**

If `--v33_calibration_consistency_weight > 0`, the module predicts a scalar `delta_calib` per view that tries to explain the residual reprojection error after triangulation.  The loss
```
L_calib = (reproj_error_view - delta_calib)^2
```
acts as a soft, learned robustifier and provides a natural auxiliary signal for the calibration-aware branch.  It is only added during training and is disabled at inference.

### 3. Integration into `OmniMultiViewFusionV5`

In `motionflow_mv/fusion/omniview_fusion_v5.py`, add the following toggles and kwargs (mirroring the v25/v30/v31 pattern):

```python
# v33 toggles
"use_camera_geometry_embedding_v33": getattr(args, "use_camera_geometry_embedding_v33", False),
"use_calibration_aware_fusion_v33": getattr(args, "use_calibration_aware_fusion_v33", False),
"v33_geom_loss_weight": getattr(args, "v33_geom_loss_weight", 0.1),
"v33_calibration_consistency_weight": getattr(args, "v33_calibration_consistency_weight", 0.0),
"v33_ray_hidden": getattr(args, "v33_ray_hidden", 64),
"v33_dropout": getattr(args, "v33_dropout", 0.1),
```

**Hook points:**

1. When `use_camera_geometry_embedding_v33=True`, instantiate `CameraGeometryEmbeddingV33` instead of `CameraConditionedViewEmbeddingV31` (which remains the default for backwards compatibility).  Feed `points_2d` and `confidences` into it so the per-joint branch has data.
2. After the v31 hierarchical encoder / v25 geometry fusion, if `use_calibration_aware_fusion_v33=True`, pass the encoder tokens and the joint geometry tokens through `CalibrationAwareFusionV33` before triangulation.
3. Add the optional calibration-consistency loss to the `geom_loss_v25` term that is already accumulated in the epipolar loss bucket.

**No existing source files are changed in this proposal**; the design is written so that v33 can be added as two new modules plus a handful of kwargs in the v5 model and training script.

## Data / preprocessing needed

* **None beyond the existing pipeline.**  The WebBridge mixed loader already provides `K, R, t` alongside `points_2d` and `confidences` (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`).
* Re-use the existing **calibration perturbation curriculum** (`motionflow_mv/calibration/camera_perturbation_curriculum.py`, invoked in `experiments/train_omniview_fusion_v5_webbridge_multi.py::apply_calibration_perturbation`) to train the calibration-aware branch on noisy cameras.
* For variable-view robustness, keep the existing `--use_variable_view_training` / `--variable_view_permute` logic; v33’s embedding is permutation-equivariant and respects `view_mask`.

## Training command / ablation flags

Recommended smoke command (local RTX 4090, based on the v32 A800 queue flags):

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
    --use_hierarchical_multiview_v31 --v31_geometry_bias \
    --use_camera_view_embedding \
    --use_camera_geometry_embedding_v33 \
    --use_calibration_aware_fusion_v33 \
    --v33_geom_loss_weight 0.1 \
    --v33_calibration_consistency_weight 0.01 \
    --v33_ray_hidden 64 \
    --v33_dropout 0.1 \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_permute \
    --cam_aug_schedule extended_curriculum \
    --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 2.0 \
    --d 64 --residual_hidden 128 --n_st_layers 2 --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 5 --early_stopping_min_delta 0.001 \
    --output outputs/v33_camera_geometry_embedding_smoke.pth
```

**Ablation flags to sweep:**

| Flag | Purpose |
|---|---|
| `--use_camera_geometry_embedding_v33` | Enable the new per-joint ray-aware embedding. |
| `--use_calibration_aware_fusion_v33` | Enable the learned geometry-bias fusion + triangulation re-weighting. |
| `--v33_geom_loss_weight W` | Weight of the auxiliary geometry consistency term. |
| `--v33_calibration_consistency_weight W` | Weight of the calibration-consistency loss (0 to disable). |
| `--v33_ray_hidden D` | Hidden dimension of the per-joint ray MLP (default 64). |
| `--v33_dropout P` | Dropout on the new MLP branches. |
| `--no_v33_calibration_consistency` | Convenience flag to disable `L_calib` while keeping the rest. |

## Expected metrics and baseline to beat

Primary evaluation is on the WebBridge H36M + MPI-INF-3DHP mixed validation split (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`).

| Metric | Baseline (v31/v32 full config) | v33 target |
|---|---|---|
| `val_MPJPE` (mm) on mixed val | ~28–32 mm for the v31/v32 full smoke/full run (exact number depends on the v32 queue outcome) | ≥ 1–2 mm improvement; close the gap toward the best single-model 9.03 mm MPI clean benchmark where possible. |
| `val_MPJPE` under variable views (k=2..14) | Current v31/v32 curve | ≥ 5% relative improvement at k≤4 and k≥10. |
| Calibration-robust val_MPJPE (rot 0.5°, focal 1%, pp 2 px) | Current calibration-curriculum baseline | ≥ 1 mm improvement on each corruption. |
| Reprojection error | Baseline from `--reproj_loss_weight 0.1` | Not degraded; ideally improved because geometry is explicit. |
| Variable-view permutation invariance | Passes existing permutation test | Maintain; the new embedding is strictly permutation-equivariant. |

Because the v32 A800 queue is still running, the concrete baseline number to beat should be taken as the **best v32 run with the same common flags** (i.e., the v32-combined or v32-ray-attention variant from `scripts/launch_v32_a800_queue.py`).  A successful v33 smoke should show a lower `val_MPJPE` than its direct v31/v32 parent on the same local 4090 config.

## Risks / unknowns

1. **Computational cost.**  The per-joint ray branch and the learned geometry bias are `O(V²·J)` in attention, the same order as the existing v31 geometry bias.  With MPI’s 14 views and `d=64` this is acceptable, but it may limit batch size on the local RTX 4090.  If smoke tests OOM, reduce `--v33_ray_hidden` or use gradient checkpointing in `CalibrationAwareFusionV33`.
2. **Overfitting to calibration noise.**  Explicitly encoding perturbed cameras can amplify the noise instead of learning robustness.  The calibration perturbation curriculum must be active (`--cam_aug_schedule extended_curriculum`) and the v33 branches must be zero-initialised so the model can ignore them at first.
3. **Interaction with physical-space losses.**  v28/v29 physical losses assume a metric 3-D space; the geometry-aware fusion should not break that.  The triangulation re-weighting is applied before the physical losses, so any change in scale must be monitored.
4. **Warm-start compatibility.**  v33 changes the shape/behaviour of the camera embedding, so a v31 checkpoint cannot be loaded directly into the v33 branch.  Two options: (a) train from scratch on top of a frozen v31 base, or (b) initialise the v33 projection to zero and load only the non-camera-embedding weights.  The proposal assumes option (b) for fast iteration.
5. **Variable-view masking edge cases.**  Because the embedding now depends on `points_2d`, masked views still have rays (but confidence=0).  The ray branch must zero-out features for masked views via `confidences`, otherwise padding views will leak geometry into the model.

## Files expected to be created (not modified)

* `docs/swarm_iter_next/v33_camera_geometry_embedding.md` — this proposal.
* `motionflow_mv/fusion/camera_geometry_embedding_v33.py` — implementation of `CameraGeometryEmbeddingV33`.
* `motionflow_mv/fusion/calibration_aware_fusion_v33.py` — implementation of `CalibrationAwareFusionV33`.
* (Subsequently, by the implementation task) `configs/v33_camera_geometry_embedding_smoke.yaml` and `scripts/launch_v33_camera_geometry_embedding_smoke.sh` for smoke/queue runs.

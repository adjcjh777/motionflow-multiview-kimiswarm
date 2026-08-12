# v33 — Contrastive Multi-View Representation Learning

**Direction slug:** `contrastive_multiview_learning`  
**Title:** Contrastive multi-view representation learning  
**Date:** 2026-08-08  
**Target model:** `OmniMultiViewFusionV5`  
**Related prior work:** `docs/swarm_iter19/20_next_iteration_synthesis.md`, `docs/swarm_iter20/synthesis.md`

---

## 1. Problem statement and motivation

The current v31/v32 stack fuses multi-view 2-D keypoints through camera-conditioned embeddings, set-transformer / Perceiver view aggregators, hierarchical multi-scale view encoders, and physical-space losses. While accuracy on full-view validation has improved steadily, the model still struggles when:

* only a small subset of views is available (variable-view inference, k=2,3,4),
* one or more views are corrupted by calibration drift or outlier observations,
* the view configuration at inference differs from the training rig (e.g. H36M 4-view → MPI-INF-3DHP 14-view transfer).

These failure modes share a common root: the learned per-view / per-joint features are not sufficiently **view-invariant**. The model relies on absolute view indices (`view_pos_embed`) and on the downstream triangulation head to compensate for view-specific biases. A contrastive objective that pulls together representations of the *same joint across different views* and pushes apart representations of *different joints* (or corrupted views) should regularise the feature space and improve generalisation under missing/corrupted views.

A legacy implementation already exists in `motionflow_mv/losses/crossview_pose_contrast.py` and was wired into an pre-v5 anchor in `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_crossview_contrast_model.py`. The v33 goal is to port and harden this idea into the current `OmniMultiViewFusionV5` architecture, making it a first-class, toggle-able component with modern ablation hooks and variable-view masking.

---

## 2. Proposed architecture changes

### 2.1 New module

Create `motionflow_mv/fusion/contrastive_multiview_v33.py` with:

* `ContrastiveMultiViewHeadV33`
  * Projects per-view per-joint ST features to a low-dimensional embedding.
  * Supports InfoNCE / NT-Xent loss with cross-view same-joint positives and different-joint negatives.
  * Handles masked-out views via a validity mask.
  * Optional “view-pooled” mode: positives are any two views observing the same joint; negatives are different joints.
  * Optional “temporal-consistent” mode: positives also include the same joint at nearby time steps (augmented by small Gaussian noise).

The existing `CrossViewJointContrastiveLoss` in `motionflow_mv/losses/crossview_pose_contrast.py` can be reused as the core loss, wrapped by the new head for feature extraction.

### 2.2 Model changes (`motionflow_mv/fusion/omniview_fusion_v5.py`)

Add to `OmniMultiViewFusionV5.__init__`:

```python
use_contrastive_multiview_v33: bool = False,
contrastive_multiview_dim: int = 64,
contrastive_multiview_temperature: float = 0.07,
contrastive_multiview_loss_weight: float = 0.1,
contrastive_multiview_mode: str = "crossview_joint",  # or "view_pooled"
contrastive_multiview_temporal_span: int = 0,         # 0 disables temporal positives
```

If `use_contrastive_multiview_v33` is True, instantiate the head and return an extra contrastive-loss tensor from `forward()` as the 6th output element.

Hook point: after the spatio-temporal (time+view) transformer and any hierarchical encoder (v30/v31), the feature tensor has shape `(B*T, V, J, d)`. This is exactly the representation consumed by the contrastive head.

### 2.3 Training-script changes (`experiments/train_omniview_fusion_v5_webbridge_multi.py`)

* Add CLI flags mirroring the model kwargs above.
* In `build_compute_loss`, consume `out[5]` when `use_contrastive_multiview_v33` is enabled and add `v33_contrastive_loss_weight * contrastive_loss` to the total loss.
* Store the raw contrastive loss in `metrics["contrastive_loss"]` for logging.

No data preprocessing changes are required; the contrastive head uses the same `view_mask` used by the ST transformer and variable-view augmentation.

### 2.4 Evaluation changes

No new eval script is required. Use the existing `experiments/eval_omniview_fusion_v5_robustness.py` with `--view_counts 2 3 4 8 14` and outlier corruption levels. The contrastive loss is active only during training and does not affect inference.

---

## 3. Training command / ablation flags

### Smoke test (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_contrastive_multiview_v33 \
  --contrastive_multiview_loss_weight 0.1 \
  --contrastive_multiview_dim 64 \
  --contrastive_multiview_temperature 0.07
```

### Full A800 run

Start from the v32 combined baseline flags and add:

```bash
  --use_contrastive_multiview_v33 \
  --contrastive_multiview_loss_weight 0.1 \
  --contrastive_multiview_dim 64 \
  --contrastive_multiview_temperature 0.07 \
  --contrastive_multiview_mode crossview_joint
```

### Ablation grid

| Run | Flags | Purpose |
|-----|-------|---------|
| `v33_contrastive_disabled` | omit `--use_contrastive_multiview_v33` | v32 baseline reproduction |
| `v33_contrastive_joint` | `--contrastive_multiview_mode crossview_joint` | full per-joint contrastive |
| `v33_contrastive_pooled` | `--contrastive_multiview_mode view_pooled` | view-pooled representation only |
| `v33_contrastive_temporal` | `--contrastive_multiview_temporal_span 2` | add temporal positives |
| `v33_contrastive_weight_0.05` | `--contrastive_multiview_loss_weight 0.05` | lower auxiliary weight |

---

## 4. Expected metrics and baseline to beat

### Primary baseline

The v32 combined run (`v32_combined` in `scripts/launch_v32_a800_queue.py`), which uses domain-aware view curriculum + trajectory consistency and is the strongest existing v32 configuration.

### Success criteria

| Condition | v32 baseline | v33 target |
|-----------|--------------|------------|
| Full-view val MPJPE (mixed H36M/MPI) | current v32 best | ≤ v32 best (no regression) |
| Variable-view MPJPE @ k=2 | ~40–60 mm | **< 35 mm** |
| Variable-view MPJPE @ k=3 | ~25–35 mm | **< 22 mm** |
| Outlier 1-view corruption (offset 25 px) | current v32 | ≤ v32 + 5% |
| Outlier 1-view corruption (offset 50 px) | current v32 | **< v32** |

If the contrastive objective is well-tuned, we expect the largest relative gain in the low-view regime, where forcing view-invariant joint features should reduce the model’s dependence on any single view.

---

## 5. Risks / unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| InfoNCE loss dominates early training and destabilises the ST feature space. | High | Start with `contrastive_multiview_loss_weight=0.05`, warm up over 3 epochs, and clamp loss at 1e3. |
| Contrastive positives are scarce under heavy occlusion / view dropout. | Medium | Use the existing `view_mask` to exclude masked joints from the denominator; fall back to joint-only negatives when k < 2. |
| Extra projection head increases memory / runtime. | Low | The head is small (d → 64 → 64) and applied only during training. |
| Temporal-positive mode may blur per-frame pose boundaries. | Medium | Keep `temporal_span=0` for first ablation; enable only after joint-only mode is stable. |
| Overfitting on H36M/MPI mixed training. | Medium | Monitor val MPJPE and use early stopping; contrastive loss is auxiliary, so disable if val regresses. |

### Open questions

1. Should the contrastive head operate on raw ST features before or after the hierarchical view encoder (v30/v31)? After may give stronger geometry-aware representations; before is cheaper and less entangled.
2. Should the loss be applied per joint, per body part, or on a globally pooled pose embedding? Per-joint is more fine-grained; part-level may be more robust for small k.
3. Is the existing `CrossViewJointContrastiveLoss` shape `(N, V, J, d)` compatible with the variable-view padding in v5, or does it need a mask-aware rewrite?

---

## 6. Deliverables and next steps

1. Implement `motionflow_mv/fusion/contrastive_multiview_v33.py` and unit test in `tests/test_contrastive_multiview_v33.py`.
2. Wire flags into `OmniMultiViewFusionV5` and `experiments/train_omniview_fusion_v5_webbridge_multi.py`.
3. Run local RTX 4090 smoke and a short d=64 full-data smoke.
4. Queue the v33 ablation on A800 after the v32 queue finishes.
5. Update this document with the first-epoch val MPJPE and contrastive-loss curves.

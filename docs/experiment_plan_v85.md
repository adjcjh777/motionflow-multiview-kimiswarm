# v85 Random View Dropout — Experiment Plan

**Status:** Implementation complete; smoke pending.  
**Date:** 2026-08-12  
**Target protocol:** H36M true-GT (S1,5,6,7,8 → S9/S11)  
**Primary goal:** Improve sparse-view (k<4) robustness by randomly dropping entire views during training and conditioning the model on the active-view count.

---

## 1. Motivation

The current best true-GT learned model, **v25 stability**, reaches **31.56 mm** on H36M S9/S11 test with 4 views.  However, variable-view evaluation shows catastrophic degradation at k=2/k=3:

| k | S9 (mm) | S11 (mm) | Notes |
|---:|---:|---:|---|
| 2 | 3017.03 | 2862.41 | learned model fails |
| 3 | 1022.08 | 1008.92 | learned model fails |
| 4 | 116.98 | 110.58 | usable |

Direct confidence-weighted DLT on the same active views achieves **58.18 / 33.32 mm** for k=2/k=3, confirming that the 2D observations are sound and the learned model is the bottleneck.  The model has been trained almost exclusively with all 4 views present, so it never learned to recover 3D poses from sparse subsets.

**v85** attacks this by making sparse-view inputs a first-class training condition:

1. **Random whole-view dropout:** during training, entire views are masked out with probability `p`, guaranteeing at least `min_views` remain.
2. **Active-view-count embedding:** the number of surviving views is injected into the ray tokens so the triangulation head can adapt its behaviour to the actual view count.
3. **Identity-at-init:** the count embedding is zero-initialised, so the module is a no-op at the start of training and preserves the v25 baseline.

---

## 2. Architecture changes

### 2.1 New module

**File:** `motionflow_mv/fusion/random_view_dropout_v85.py`

```text
RandomViewDropoutV85
├── apply_dropout(view_mask) → view_mask
│       Training: sample keep mask ~ Bernoulli(1 − p)
│       Enforce min_views via deterministic fallback
│       Return updated boolean view mask
│       Inference: no-op
└── embed_tokens(tokens, view_mask) → tokens
        active = sum(view_mask, dim=-1)   # (B, T)
        tokens += count_embed(active)       # zero-init embedding
```

Key hyperparameters:

| Flag | Default | Meaning |
|------|------:|---------|
| `use_random_view_dropout_v85` | `False` | Enable the module |
| `v85_dropout_prob` | `0.3` | Per-view dropout probability during training |
| `v85_min_views` | `2` | Minimum retained views after dropout |
| `v85_use_count_embedding` | `True` | Inject active-view-count embedding into ray tokens |

### 2.2 Integration into v25 geometry fusion

**File:** `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`

The module is inserted **after** v45 adaptive geometry fusion / v84 uncertainty-weighted dropout and **before** the ray tokenizer:

```python
# Optional v85 random view dropout on the view mask.
if self.use_random_view_dropout_v85 and self.random_view_dropout_v85 is not None:
    if view_mask is None:
        view_mask = torch.ones(B, T, V, device=pts.device, dtype=torch.bool)
    view_mask = self.random_view_dropout_v85.apply_dropout(view_mask)
    confidence = confidence * view_mask[:, :, :, None].float()

# World rays and ray tokens.
centre, direction = compute_rays(pts, K, R, t)
tokens = self.ray_tokenizer(centre, direction, confidence)

# Optional v85 active-view-count embedding.
if self.use_random_view_dropout_v85 and self.random_view_dropout_v85 is not None:
    tokens = self.random_view_dropout_v85.embed_tokens(tokens, view_mask)
```

This guarantees that:
- Dropped views contribute zero confidence to the ray tokenizer.
- The count embedding is available to downstream v83 (view-conditioned temporal attention) and v84 (uncertainty-weighted dropout) modules if they are enabled.

### 2.3 Flag plumbing

- `motionflow_mv/fusion/omniview_fusion_v5.py` accepts the four v85 hyperparameters and forwards them to `MultiViewGeometryFusionV25`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` exposes the CLI flags and adds them to the model-kwargs dictionary.

---

## 3. Training configuration

### 3.1 Base recipe

v85 builds on the **v25 stability** recipe (low LR, no view permutation).  The only additions are the v85 flags and a slightly higher base dropout to compensate for the extra regularisation.

### 3.2 A800 medium run

**Script:** `scripts/run_v85_random_view_dropout_medium_a800.sh`

| Setting | Value |
|---|---|
| GPU | 7 (project default; only GPUs 6/7 are allowed on A800) |
| Epochs | 20 |
| Batch size | 16 |
| Train samples | 4096 |
| Clip length | 13 |
| LR | 1e-4 |
| Warmup | 4 epochs |
| Grad norm clip | 1.0 |
| EMA decay | 0.999 |
| Early stopping | patience 3, delta 0.001 |
| v85 dropout prob | 0.3 |
| v85 min views | 2 |
| Count embedding | enabled |

Output files:

| File | Path |
|---|---|
| Checkpoint | `outputs/ablations/v85_random_view_dropout_medium_a800.pth` |
| Config | `outputs/ablations/v85_random_view_dropout_medium_a800.config.json` |
| Log | `outputs/ablations/v85_random_view_dropout_medium_a800.log` |

### 3.3 Local RTX 4090 smoke

A 2-epoch smoke is recommended before launching the A800 run:

```bash
python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --num_domains 1 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_random_view_dropout_v85 \
    --v85_dropout_prob 0.3 \
    --v85_min_views 2 \
    --v85_use_count_embedding \
    --clip_len 9 \
    --epochs 2 \
    --batch_size 4 \
    --train_samples 256 \
    --lr 1e-4 \
    ...
```

Expected smoke outcome: val MPJPE should be finite and within ~5 mm of the v25 stability baseline by epoch 2.  If divergence is observed, lower `v85_dropout_prob` to `0.2` or `0.1`.

---

## 4. Expected evaluation protocol

### 4.1 Full-view test (k=4)

Evaluate the best checkpoint on the standard H36M true-GT S9/S11 test set:

```bash
python experiments/eval_omniview_fusion_v5.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v85_random_view_dropout_medium_a800.pth \
    --config outputs/ablations/v85_random_view_dropout_medium_a800.config.json \
    --dataset_manifest configs/splits/h36m_true_gt_standard.yaml \
    --eval_subjects S9 S11 \
    --output outputs/eval_v85_true_gt_h36m_test.json
```

**Primary gate:** test MPJPE < 31.56 mm (beats v25 stability).  
**Robustness gate:** PA-MPJPE < 34.35 mm (beats v25 stability PA-MPJPE).

### 4.2 Variable-view test (k=2/3/4)

Run the companion variable-view script:

```bash
bash scripts/eval_variable_views_v85_random_view_dropout_medium_a800.sh
```

Expected improvements:

| k | S9 target | S11 target | Notes |
|---:|---:|---:|---|
| 2 | < 300 mm | < 300 mm | first non-catastrophic result |
| 3 | < 150 mm | < 150 mm | close to DLT-fallback k=3 |
| 4 | < 110 mm | < 110 mm | preserve full-view quality |

The long-term goal is to match or beat direct DLT on k=2/k=3 (58.18 / 49.35 mm for S9, 33.32 / 25.28 mm for S11).  If v85 alone does not reach this, combine with the v25 DLT-fallback wrapper.

---

## 5. How to run the scripts

### 5.1 Training on A800 GPU 7

```bash
# From the A800 repo root
bash scripts/run_v85_random_view_dropout_medium_a800.sh
```

Optional: run on a different GPU:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_v85_random_view_dropout_medium_a800.sh
```

### 5.2 Monitor training

```bash
ssh a800-D "tail -40 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"
```

### 5.3 Variable-view evaluation

```bash
bash scripts/eval_variable_views_v85_random_view_dropout_medium_a800.sh
```

Results will be written to:

- `outputs/variable_view_v85_random_view_dropout_medium_a800.csv`
- `outputs/variable_view_v85_random_view_dropout_medium_a800.json`
- `outputs/variable_view_v85_random_view_dropout_medium_a800.log`

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Training diverges due to aggressive dropout | Start with `v85_dropout_prob=0.2`; only increase if stable |
| Full-view k=4 quality degrades | Count embedding is zero-init and residual; dropout disabled at eval |
| No sparse-view improvement | Combine with v25 DLT-fallback wrapper or increase `train_samples` |
| Too many dropped samples have only min_views | `min_views=2` is already the sparse target; use data augmentation (outlier views) alongside |

---

## 7. Next steps after v85

1. Run the local RTX 4090 smoke (2 epochs).  If val MPJPE is stable, launch the A800 medium run.
2. After training, run full-view S9/S11 test and variable-view k=2/3/4 evaluation.
3. If v85 improves k<4 without hurting k=4, consider combining with the v25 DLT-fallback wrapper for a production sparse-view inference path.
4. Record results in `docs/results_true_gt_h36m.md` and update `AGENTS.md`.

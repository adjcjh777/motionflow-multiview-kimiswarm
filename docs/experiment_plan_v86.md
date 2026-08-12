# v86 No Count Embedding — Experiment Plan

**Status:** Plan ready; launch pending free A800 GPU.  
**Date:** 2026-08-12  
**Target protocol:** H36M true-GT (S1,5,6,7,8 → S9/S11)  
**Primary goal:** Isolate the contribution of the v85 active-view-count embedding by disabling it while keeping the random view-dropout augmentation.

---

## 1. Motivation

**v85** combines two mechanisms to improve sparse-view (k<4) robustness:

1. **Random whole-view dropout** during training.
2. **Active-view-count embedding** injected into the ray tokens.

It is not obvious how much of the (hoped-for) sparse-view improvement comes from the dropout itself (forcing the model to see k=2/3/4 inputs) versus the count embedding (telling the triangulation head how many views are active).  **v86** ablates the count embedding so we can compare:

| Variant | Dropout | Count embedding | Purpose |
|--------:|---------|-----------------|---------|
| v25 stability | No | No | Baseline |
| v85 | Yes | Yes | Full v85 treatment |
| **v86** | **Yes** | **No** | **Ablated: isolate count-embedding contribution** |

If v86 still improves over v25 stability for k<4, then dropout alone is useful.  If v86 is worse than v85, the count embedding adds value beyond the augmentation.

---

## 2. Architecture / flag change

The only difference from v85 is the removal of the active-view-count embedding.  In `motionflow_mv/fusion/random_view_dropout_v85.py` this corresponds to:

```python
RandomViewDropoutV85(
    ...,
    use_count_embedding=False,  # v86: disabled
)
```

At the CLI this is expressed by replacing v85's:

```bash
--v85_use_count_embedding
```

with:

```bash
--no_v85_use_count_embedding
```

All other v85 hyperparameters remain identical:

| Flag | Value | Meaning |
|------|------:|---------|
| `use_random_view_dropout_v85` | `True` | Enable random whole-view dropout |
| `v85_dropout_prob` | `0.3` | Per-view dropout probability during training |
| `v85_min_views` | `2` | Minimum retained views after dropout |
| `v85_use_count_embedding` | `False` | **Disabled for v86** |

---

## 3. Training configuration

### 3.1 Base recipe

Identical to the v85 medium run except for the disabled count embedding.  Builds on the **v25 stability** recipe (low LR, no view permutation) with v85 dropout enabled.

### 3.2 A800 medium run

**Scripts:**
- `scripts/run_v86_no_count_embedding_medium_a800_gpuX.sh` (default GPU 7)
- `scripts/run_v86_no_count_embedding_medium_a800_gpu6.sh` (default GPU 6)

| Setting | Value |
|---|---|
| GPU | 6/7 (default GPU 6 or 7, override with `CUDA_VISIBLE_DEVICES`) |
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
| Count embedding | **disabled** |

**GPU policy note:** Project policy restricts MotionFlow to GPU 6 and GPU 7 only; GPUs 0–5 are reserved. The launch scripts default to GPU 6/7 accordingly.

Output files:

| File | Path |
|---|---|
| Checkpoint | `outputs/ablations/v86_no_count_embedding_medium_a800.pth` |
| Config | `outputs/ablations/v86_no_count_embedding_medium_a800.config.json` |
| Log | `outputs/ablations/v86_no_count_embedding_medium_a800.log` |

### 3.3 Launch command

```bash
# Default: GPU 7 (only GPU 6/7 may be used by this project)
bash scripts/run_v86_no_count_embedding_medium_a800_gpuX.sh

# Default: GPU 6
bash scripts/run_v86_no_count_embedding_medium_a800_gpu6.sh

# Run on a different free GPU (must be 6 or 7)
CUDA_VISIBLE_DEVICES=6 bash scripts/run_v86_no_count_embedding_medium_a800_gpuX.sh
```

**Scheduling status (2026-08-12):** v85 random-view-dropout training is currently running on GPU 7 and a no-fallback variable-view eval is running on GPU 6. v86 must wait for one of those GPUs to become free before launching.

---

## 4. Expected evaluation protocol

### 4.1 Full-view test (k=4)

Evaluate the best checkpoint on the standard H36M true-GT S9/S11 test set:

```bash
python experiments/eval_omniview_fusion_v5.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v86_no_count_embedding_medium_a800.pth \
    --config outputs/ablations/v86_no_count_embedding_medium_a800.config.json \
    --dataset_manifest configs/splits/h36m_true_gt_standard.yaml \
    --eval_subjects S9 S11 \
    --output outputs/eval_v86_no_count_embedding_true_gt_h36m_test.json
```

**Primary gate:** test MPJPE should remain competitive with v25 stability (**31.56 mm**) and v85.  
**Robustness gate:** PA-MPJPE should remain near v25 stability (**34.35 mm**) and v85.

### 4.2 Variable-view test (k=2/3/4)

Create or adapt a variable-view evaluation script that mirrors `scripts/eval_variable_views_v85_random_view_dropout_medium_a800.sh` but points to the v86 checkpoint.  Run with `--var_view_dlt_fallback` so k<4 results that are still catastrophic fall back to direct confidence-weighted DLT.

Expected comparison matrix:

| Variant | k=2 S9/S11 | k=3 S9/S11 | k=4 S9/S11 | Notes |
|---------|-----------:|-----------:|-----------:|-------|
| v25 stability | ~3017 / ~2862 mm | ~1022 / ~1009 mm | 116.98 / 110.58 mm | learned model fails for k<4 |
| v25 + DLT fallback | 58.18 / 49.35 mm | 33.32 / 25.28 mm | 116.98 / 110.58 mm | direct DLT for k<4 |
| v85 | TBD | TBD | TBD | dropout + count embedding |
| **v86** | **TBD** | **TBD** | **TBD** | **dropout only** |

**Analysis question:** does v86 close most of the k<4 gap relative to v85, or does the count embedding carry most of the benefit?

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Without count embedding, sparse-view inputs drift because the model has no explicit signal of view count | Monitor k=2/k=3 variable-view MPJPE; if worse than v25 + DLT fallback, the count embedding is essential |
| Full-view k=4 degrades because dropout removes a useful conditioning signal | Count embedding is zero-init and residual; its absence should not hurt k=4 if dropout is stable |
| Dropout alone is not enough to learn sparse-view triangulation | Combine with `--var_view_dlt_fallback` at inference as a production fallback |

---

## 6. Next steps after v86

1. Launch v86 on a free A800 GPU (GPU 6 or 7 only) once v85 training/eval finishes.
2. After training, run full-view S9/S11 test and variable-view k=2/3/4 evaluation with DLT fallback.
3. Compare v86 directly with v85 and v25 stability:
   - If v86 ≈ v85, the count embedding is redundant and can be removed to save parameters.
   - If v86 << v85, the count embedding is a necessary component.
   - If v86 is worse than v25 + DLT fallback, then dropout alone is harmful without the count embedding.
4. Record results in `docs/results_true_gt_h36m.md` and update `AGENTS.md`.

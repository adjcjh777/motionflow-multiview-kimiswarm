# Agent-01: v45-AGF Medium Local RTX 4090 Status Analysis

**Run:** `scripts/run_v45_agf_medium_local_4090.sh`  
**Log:** `outputs/v45_agf_medium_local_4090.log`  
**Status:** Still running (process alive as of analysis)  
**Tracking:** #160, depends on #154 / v45-AGF

## Configuration snapshot

| Setting | Value |
|---|---|
| Epochs | 5 |
| `train_samples` | 500 |
| `batch_size` | 4 |
| `clip_len` | 9 |
| `d` | 64 |
| `residual_hidden` | 128 |
| `n_st_layers` | 2 |
| Adaptive weight type | `per_view` |
| Variable-view training | enabled (2–8 views, curriculum start 4) |
| Hierarchical multi-view (v30) | enabled |
| LR | 1e-3 cosine, 1-epoch warmup |
| Early stopping | patience 3, min delta 0.001 |

## Current progress

As of the latest log tail, the run is **in the middle of Epoch 2**.

- **Epoch 1 completed:**
  - `train_loss` ended at **5.843691**
  - `val_loss` = 0.029085
  - `val_MPJPE` = **31.95 mm**
- **Epoch 2 in progress:**
  - Step 50 loss: 7.610562
  - Step 1500 loss: ~7.362
  - Loss has plateaued around **7.35–7.45** after an initial post-epoch jump.

## Loss/val trend analysis

### Epoch 1 training loss curve

| Step range | Loss behavior |
|---|---|
| 0–500 | Steep decline from 20.63 → ~8.69 |
| 500–1500 | Smooth decline from ~8.4 → ~6.5 |
| 1500–3600 | Slow convergence from ~6.5 → **5.84** |

The first epoch shows healthy, monotonic convergence with no spikes or NaNs. The model learned quickly in the first third of the epoch and then fine-tuned.

### Epoch 2 training loss curve

- The loss jumps to **7.61** at the start of Epoch 2 (common when data shuffling / curriculum / augmentation resets).
- It then slowly decreases but has flattened around **7.35–7.40** by step 1500.
- The decline rate is much slower than Epoch 1, suggesting the model is approaching a local minimum for the training objective.

## Prediction for final validation MPJPE

**Point estimate: 25–29 mm best `val_MPJPE`, likely reached between Epoch 3 and Epoch 4.**

Rationale:

1. Epoch 1 already achieved **31.95 mm** with only one epoch of training on the medium 500-sample manifest. Comparable local runs:
   - v25 small local (500 samples): 63.13 mm
   - v42 local (d=64, similar scale): 26.16 mm
   - v34 VJGN / v36 local: ~26–27 mm
2. The Epoch 2 training loss plateau near 7.36 suggests the model is close to convergence; further epochs will yield diminishing but still positive gains.
3. With 5 epochs and early-stopping patience of 3, the best checkpoint is likely to occur before Epoch 5 unless overfitting begins.
4. A regression at full views is not indicated by the current trend; losses are stable and no divergence is visible.

### Scenario table

| Scenario | Final best val_MPJPE | Trigger |
|---|---|---|
| Optimistic | ~23–25 mm | Continued slow improvement through Epoch 4 |
| Base case | ~26–29 mm | Best around Epoch 3–4, then plateau |
| Pessimistic | ~30–33 mm | Early overfitting or curriculum/augmentation noise dominates |

## Implications for v46-SVG

- v45-AGF appears to be training stably on variable views, which is a positive signal that the existing `variable_view_training` path can serve as the foundation for v46 sparse-view generalization.
- The Epoch 1 val_MPJPE of 31.95 mm is already within the v46 smoke target (< 80 mm), so v46 should be able to build on this base.
- If the final best val_MPJPE lands below 28 mm, v46-SVG should use v45-AGF as the baseline rather than reverting to v25.

## Blockers / open questions

- The log is live; the run had not completed Epoch 2 at the time of analysis. A follow-up report after Epoch 3–5 would refine the prediction.
- No GPU OOM or NaN observed.

# v30 Design Proposal: Stable Hierarchical Physical Multi-View Pose Estimation

## Motivation (from v29 findings)

v29 explored three new components for the SEH-MV pipeline:

1. **Hierarchical multi-scale view encoder** (`HierarchicalViewEncoderV29`) – adds joint/part/body cross-view attention.
2. **Test-time self-evolution (TTE)** – iterative geometric view re-weighting at inference.
3. **Physical-space temporal loss** – training-time priors for floor, bone-length smoothness, and COM jitter.

### Results

| Variant | val_MPJPE (epoch 1) | Notes |
|---------|----------------------|-------|
| v29a (hierarchical only, d=64) | 28.12mm | Overfits to 47.85mm by epoch 2 |
| v29b (hierarchical + TTE) | 90.35mm | TTE catastrophically degrades results |
| v29d (TTE + physical) | 90.28mm | Same TTE failure |
| v25 small baseline (A800) | ~18mm | Still the best small-config result |

Conclusions:
- **TTE is broken** and must be disabled/defanged before any inference-time refinement is trusted.
- **Hierarchical encoder shows promise** but is worse than v25 and overfits quickly on small data.
- **Physical loss is untested** in isolation; the 90mm numbers are dominated by TTE, not physical loss.

## v30 Direction

Build a **stable hierarchical physical multi-view pose estimator** by:

1. Keeping the hierarchical view encoder idea but hardening it against overfitting.
2. Using the physical temporal loss as a training regularizer from the start.
3. Removing TTE from the main pipeline until it is proven safe.
4. Scaling up training data and model capacity where the ablations indicate benefit.

### Proposed architecture changes

#### 1. Stable Hierarchical View Encoder (v30)

Replace `HierarchicalViewEncoderV29` with `HierarchicalViewEncoderV30`:

- **Learned part grouping** (or at least dataset-aware part groups) instead of hard-coded H36M/MPI groups.
- **Cross-scale residual fusion**: each scale receives residuals from coarser scales before cross-view attention.
- **Stochastic depth / dropout** on attention layers to reduce overfitting.
- **LayerNorm + scaled residual** for stable training at d=128 and n_st_layers=3.
- **Identity-at-init** is kept, but the residual path is gated so the block is zero at initialization.

#### 2. Physical Temporal Loss as First-Class Regularizer

Integrate `PhysicalSpaceTemporalLossV29` into training with:
- Warmup schedule for physical weights (start at 0, ramp up over first few epochs).
- Dataset-aware parents and foot-joint indices.
- Optional: bone-length prior as a soft constraint rather than a temporal smoothness term.

#### 3. Remove TTE from the paper story (for now)

- TTE may be revisited as a post-hoc refinement once the coordinate-frame / scale bug is fully understood.
- For v30, the "self-evolution" comes from the **hierarchical structure** and **physical priors**, not from test-time iteration.

### Training recipe

- Datasets: WebBridge + H36M + MPI mixed loader.
- Model: d=128, residual_hidden=256, n_st_layers=3, n_part_layers=2.
- Physical weights: floor=0.01, bone_temporal=0.01, com_jitter=0.001 (with warmup).
- Data: train_samples=4000–8000, clip_len=243 (or full sequence), batch_size=16–24.
- Augmentation: variable views (2–14), outlier views, camera dropout.
- Optimizer: lr=1e-3, cosine, warmup 3 epochs, max_grad_norm=1.0, ema_decay=0.999.
- Early stopping patience=5 (longer than v29's 3 to allow physical loss to stabilize).

### Evaluation

- val_MPJPE and PA-MPJPE on H36M/MPI validation.
- Few-view robustness: 2, 4, 8 views.
- Cross-dataset generalization: train on WebBridge+H36M, test on MPI.
- Ablation: hierarchical vs. physical vs. both.

## Go/No-Go Criteria

**Proceed to v30 implementation if:**
- Hierarchical-only at d=128 or n_st_layers=3 closes the gap to v25 small baseline.
- Physical loss shows clear regularization benefit (lower overfitting, comparable or better val).

**Pivot to redesign if:**
- Hierarchical variants remain ≥30mm at d=128.
- Physical loss harms performance at all weights.

## Next Immediate Steps

1. Wait for the redesigned v29 sweep (hierarchical-only + physical-only) to finish first epochs.
2. Pick the best hierarchical configuration and scale it.
3. Implement `HierarchicalViewEncoderV30` with the hardening changes.
4. Run v30 full-scale on A800.

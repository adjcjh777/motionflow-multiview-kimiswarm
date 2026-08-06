# Iter11+ Advanced Model Architecture Report

**Scope:** `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py` and surrounding training/evaluation pipeline.  
**Goal:** identify concrete, implementable changes that push MPI-INF-3DHP validation MPJPE below the current ~11.17 mm cross-view residual baseline and strengthen paper quality for ICRA/CVPR 2027.

## 1. Current State

The new advanced model (`RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`) fuses five prior ideas:

1. Per-view 2D keypoint + ray encoding with camera-conditioned embeddings.
2. Joint-level and view-level attention (from `RayAttentionFusionModelTemporalCrossview`).
3. A single spatio-temporal transformer that flattens the `(time, view)` grid per joint.
4. A Gaussian uncertainty head that predicts per-view log-variance and converts it to DLT weights.
5. A differentiable Gauss-Newton (GN) solver that refines the weighted-DLT estimate, followed by a residual MLP.

Training is currently done with a small script that uses a fixed learning-rate Adam optimizer, light augmentation, and limited data. A fast run reached 47.54 mm because of insufficient epochs/data, while the cross-view residual model reaches ~11.17 mm. The advanced model is therefore promising but under-optimized.

## 2. Proposed Improvements

### 2.1 Factorize Spatio-Temporal Attention

The current implementation flattens `T * V` tokens and runs a single transformer layer. This is expensive, loses the structural separation between temporal and view axes, and generalizes poorly when `T` or `V` changes.

**Action:** replace the flat `T*V` self-attention with factorized blocks that alternate view-only and time-only attention. Keep the `(B*J, T*V, d)` path for minimal invasiveness.

```python
# Pseudo-code for factorized attention
for block in self.st_blocks:
    feat = feat.view(B, J, T, V, d)
    feat = self.view_attn(feat)      # attend across views
    feat = self.temporal_attn(feat)  # attend across time
    feat = feat.view(B*J, T*V, d)
```

Expected gain: ~5–10% lower MPJPE and faster training on longer clips.

### 2.2 Robust Gauss-Newton with Learned Damping

The GN solver uses a fixed scalar damping term and a squared loss, so it is not robust to outliers.

**Action:**

- Add a small MLP that predicts per-joint damping from pooled features.
- Use a Geman-McClure or Huber kernel inside GN to down-weight large reprojection residuals.
- Optionally gate out views whose uncertainty exceeds a learned threshold.

```python
# Pseudo-code for robust GN
weights_robust = weights * robust_kernel(r, c=5.0)
damping = self.damping_mlp(feat_pooled).squeeze(-1)  # (N, J)
A, b = build_normal_equations(J_world, r, weights_robust, damping)
dx = torch.linalg.solve(A, b)
```

This directly targets the long-tail errors that dominate PCK@150mm.

### 2.3 Skeleton-Aware Residual Refinement

The residual MLP is per-joint and ignores anatomical constraints.

**Action:** insert a lightweight graph attention layer over the skeleton adjacency matrix inside the residual head.

```python
# Pseudo-code for skeleton-aware residual head
class SkeletonAwareResidual(nn.Module):
    def __init__(self, d, residual_hidden, adjacency):
        super().__init__()
        self.mlp = nn.Sequential(...)  # (d+3) -> residual_hidden
        self.gat = GATLayer(residual_hidden, residual_hidden, adjacency)
        self.out = nn.Linear(residual_hidden, 3)

    def forward(self, feat, pred_3d):
        h = self.mlp(torch.cat([feat, pred_3d], dim=-1))
        h = self.gat(h)
        return pred_3d + self.out(h)
```

Expected gain: reduced wrist/ankle jitter and lower PA-MPJPE.

### 2.4 Auxiliary Bone-Length and Temporal-Smoothness Losses

Current training uses MSE plus optional reprojection. Add anatomical and temporal regularizers:

```python
def bone_length_loss(pred, gt, skeleton):
    bl_pred = pred[:, :, parent] - pred[:, :, child]
    bl_gt   = gt[:, :, parent] - gt[:, :, child]
    return F.mse_loss(bl_pred.norm(dim=-1), bl_gt.norm(dim=-1))

def temporal_smoothness_loss(pred):
    return (pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2]).pow(2).mean()
```

### 2.5 Training Regime Upgrades

The fast run was limited by data/epochs. To reach publishable numbers:

1. **Scale data:** train on all MPI-INF-3DHP subjects/sequences and the in-progress H36M WebBridge conversion (`data/webbridge/h36m`).
2. **Stronger augmentation:** increase outlier/occlusion rates, add camera-jitter, and use correlated view dropout.
3. **Optimizer:** AdamW with cosine decay, linear warmup, gradient clipping, and Exponential Moving Average (EMA).
4. **Efficiency:** enable AMP and `DistributedDataParallel`.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
scaler = torch.cuda.amp.GradScaler()
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
```

## 3. Recommended Experiments

| Experiment | What to vary | Primary metric |
|------------|--------------|----------------|
| Factorized vs. flat ST attention | `view_attn + temporal_attn` vs. `st_transformer` | MPJPE, training time |
| Robust GN kernel | Huber / Geman-McClure / squared | MPJPE, PCK@150mm |
| Skeleton-aware residual | GAT vs. MLP head | PA-MPJPE, per-joint wrist/ankle error |
| Data scale | 1 vs. 6 MPI-INF-3DHP train sequences | MPJPE |
| H36M pre-training | train on H36M, finetune on MPI-INF-3DHP | MPI-INF-3DHP MPJPE |
| Uncertainty loss weight | 0.0, 0.05, 0.1, 0.2 | MPJPE + calibration |
| Model capacity | d in {64,128}, n_st_layers in {2,3,4} | MPJPE vs. params |

## 4. Metrics to Track

- **MPJPE** and **PA-MPJPE** (primary accuracy).
- **PCK@50/100/150 mm** and **AUC** (standard 3DHP benchmarks).
- **Per-joint MPJPE** to find systematic failures (wrists/ankles).
- **View-drop robustness:** MPJPE when one view is randomly removed.
- **Uncertainty calibration:** correlation between predicted variance and actual reprojection error.
- **Efficiency:** wall-clock time per epoch, memory (GB), parameter count.

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Combining all modules makes training unstable | Use skip connections, layer normalization before triangulation, initialize from cross-view-residual checkpoint |
| Robust GN kernel hurts gradients | Use Huber with large transition, or add after warmup |
| Graph residual head increases memory | Use sparse adjacency and small GAT hidden size |
| H36M WebBridge conversion unfinished | Finish `data/webbridge/h36m` before large transfer runs |
| Overfitting with larger model | EMA, weight decay, stronger augmentation, early stopping |

## 6. Immediate Next Steps

1. Implement factorized attention in a new `...V2` model class to preserve the v1 baseline.
2. Add robust GN and skeleton-aware residual as optional flags for easy ablation.
3. Run a 30-epoch ablation with AdamW, EMA, stronger augmentation, and the full MPI-INF-3DHP train set.
4. Once H36M WebBridge conversion is complete, train a cross-dataset model and evaluate zero-shot on MPI-INF-3DHP.
5. Re-evaluate the current best cross-view-residual checkpoint (11.17 mm) as the baseline for all new experiments.

---

*Prepared for the MotionFlow-MultiView ICRA/CVPR 2027 submission roadmap, Iter11+.*

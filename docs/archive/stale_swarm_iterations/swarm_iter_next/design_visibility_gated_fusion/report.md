# Explicit Visibility-Gated Fusion for Occluded Views

## Summary

This task adds an explicit visibility-gating branch to the current best
`RayAttentionFusionModelTemporalResidual` architecture.  Instead of relying only
on implicit per-view soft weights and detector confidence, the new
`VisibilityGatedFusionModel` predicts a per-view, per-joint soft visibility
mask and multiplies the triangulation weights by it.  A fallback guard keeps the
triangulation stable when too many views are occluded, and an auxiliary BCE
loss on synthetic occlusion labels supervises the visibility head.

## Files Added / Modified

| File | Purpose |
|------|---------|
| `motionflow_mv/fusion/visibility_gated_fusion.py` | `VisibilityGatedFusionModel`, `VisibilityGatedFusionModule`, and sanity checks. |
| `experiments/train_visibility_gated_mpiinf3dhp.py` | Training script with occlusion-aware loss, fallback guard, and a `--smoke` mode that runs on synthetic data. |
| `docs/swarm_iter_next/design_visibility_gated_fusion/report.md` | This report. |

No existing files were modified; the model subclasses the current best
architecture and the trainer reuses the shared `TemporalClipDataset` and
`RandomClipDataset`.

## Architecture

```text
input (B,T,V,J,3)
  |
  +-> per-frame encoder (view + joint attention) ----+ (B*T,V,J,d)
  |                                                   |
  +-> temporal transformer ----------------------------+
                                                       |
                                            visibility_head (d -> 1)
                                                       |
                                            v_logits (B*T,V,J)
                                                       |
                                            visibility = sigmoid(v_logits)
                                                       |
                                            fallback guard
                                                       |
                                            weights *= confidences * visibility
                                                       |
                                            differentiable weighted DLT
                                                       |
                                            residual refinement head
                                                       |
                                            pred_3d, weights, v_logits
```

### Visibility head

A small MLP on each `(view, joint)` token after the temporal transformer:

```python
self.visibility_head = nn.Sequential(
    nn.Linear(d, d // 2),
    nn.ReLU(),
    nn.Linear(d // 2, 1),
)
```

It predicts a scalar logit per view/joint; `visibility = sigmoid(logit)` is
then multiplied into the DLT weights.

### Fallback guard

If fewer than `min_visible_views` are predicted visible for a given joint, the
model temporarily treats all views as visible for that joint to avoid a degenerate
DLT system:

```python
visible = (visibility > self.visibility_threshold).float()
visible_count = visible.sum(dim=1)
fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)
effective_visibility = visibility + (1.0 - visibility) * fallback
```

### Training loss

```
L = MSE(pred_3d, y_gt)
    + λ_occ · BCE_with_logits(v_logits, visible_target)
    + optional reproj / bone-length terms
```

The visibility target is generated from the existing dropout augmentation: any
detection whose confidence is zeroed by dropout is labelled occluded
(`visible_target = 0`).

## Usage

### Smoke test (CPU, no real data)

```bash
python experiments/train_visibility_gated_mpiinf3dhp.py --smoke
```

This generates a 120-frame synthetic training clip and a 40-frame validation clip,
then trains for two epochs on CPU.

### Forward/gradient sanity check

```bash
python motionflow_mv/fusion/visibility_gated_fusion.py
```

### Real training (once data is available)

```bash
python experiments/train_visibility_gated_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --epochs 30 --occlusion_loss_weight 0.1
```

## Expected Impact

- **Robustness to occlusion:** Explicit visibility masking should improve
  MPJPE under moderate-to-heavy occlusion by suppressing corrupted views.
- **Interpretability:** The model now returns per-view visibility logits that
  can be inspected to understand why a view was ignored.
- **No regression risk:** Because the visibility head is multiplicative and the
  fallback guard prevents degenerate triangulation, clean-view accuracy should
  remain close to the 10.46 mm baseline.
- **Auxiliary loss:** The BCE term is cheap and only requires the existing
  dropout augmentation labels.

## Verification Plan

1. Shape/gradient sanity check via `__main__` block and smoke training.
2. Quick 2-epoch smoke test on synthetic data to confirm the loss converges
   without NaNs and the checkpoint is saved.
3. Full MPI-INF-3DHP S1→S2/Seq1 training once WebBridge data is available.
4. Robustness sweep with injected per-joint occlusion rates
   (`0%, 10%, 30%, 50%`) comparing visibility-gated vs. baseline residual model.

## Risks / Blockers

- **Synthetic occlusion labels:** Random dropout is only a coarse proxy for real
  occlusion.  Real gains may require pose-dependent or view-correlated occlusion
  simulation (e.g., using MPI-INF-3DHP confidence thresholds or body-part
  self-occlusion maps).
- **Fallback guard gradient:** The hard fallback is non-differentiable but only
  activates for rare degenerate cases; in practice this is acceptable for a
  prototype.
- **Visibility head under-confidence:** If the BCE weight is too high the model
  may learn to mark most views as occluded.  Start with `λ_occ = 0.1` and tune.
- **Data availability:** Smoke tests do not need real data, but the final MPI
  evaluation requires the WebBridge canonical `.npz` files.

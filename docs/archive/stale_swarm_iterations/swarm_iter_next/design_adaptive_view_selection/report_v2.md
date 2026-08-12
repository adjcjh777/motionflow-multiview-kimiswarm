# Adaptive Soft View Gate v2

**Swarm task:** task_10 — Improve adaptive view selection soft gate  
**Author:** sub-agent task_10  
**Date:** 2026-08-05

## Summary

This report documents the v2 improvement of the adaptive soft view gate in `RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGate`. The v1 gate applies a per-view MLP to the encoder tokens and thresholds the score with a sigmoid. v2 replaces this isolated scorer with a **cross-view attention scorer** that is also **confidence- and ray-geometry aware**, and uses a **learnable sigmoid temperature**.

## Motivation

In v1 each view is scored independently. This has two limitations:

1. **No view-to-view comparison.** A view cannot directly attend to the other views to decide whether it is redundant, occluded, or geometrically weak.
2. **Fixed softness.** The sigmoid temperature is implicit and hard-coded through the MLP scale, making it hard to tune the selectivity of the gate.

The v2 gate addresses both issues.

## Architecture

### File

`motionflow_mv/fusion/ray_attention_temporal_residual_campe_adaptive_softgate_v2_model.py`

### New module: `SoftViewGateV2`

The gate lives inside `RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGateV2` and is inserted after the temporal encoder, just before triangulation.

**Inputs:**

- `feat`: `(N, V, J, d)` temporal encoder tokens, where `N = B * T`.
- `points_2d`: `(N, V, J, 2)` 2D keypoints.
- `K, R, t`: camera parameters.
- `confidences`: `(N, V, J)` input detection confidences.

**Per-view score token:**

```
token_v^j = [ feat_v^j || confidence_v^j || ray_v^j ]
```

where `ray_v^j` is the world-space ray direction (3-D). The token dimension is `d + 4`.

**Cross-view attention:**

Tokens are reshaped to `(N*J, V, d+4)` and passed through a `nn.MultiheadAttention` layer. This lets each view score itself relative to the others.

**Score and gate:**

```
score_v^j = MLP(Attention(token_v^j))
gate_v^j = sigmoid(score_v^j / tau)
```

The temperature `tau = softplus(log_tau) + 1e-3` is learned end-of-end, allowing the gate to become sharper or softer during training.

**Triangulation:**

The existing per-view DLT weights are multiplied by the soft gate:

```
weights_v^j = weight_head(feat)_v^j * confidence_v^j * gate_v^j
```

### Regularisation

Three terms are combined into a scalar `reg` returned by the model:

1. **Budget loss:** encourage the average number of selected views to be `target_k`.
   ```
   L_budget = mean_v (mean_v gate_v^j - target_k / V)^2
   ```
2. **Minimum-views loss:** penalise selections below `min_views`.
   ```
   L_min = ReLU(min_views - sum_v gate_v^j).mean()
   ```
3. **Sharpness loss:** encourage gate values to spread apart (sharper decisions).
   ```
   L_sharp = -std_v(gate_v^j).mean()
   ```

The total regulariser is `reg = L_budget + L_min + 0.1 * L_sharp`.

## Differences from v1

| Aspect | v1 | v2 |
|--------|-----|-----|
| Score mechanism | Per-view MLP on `feat` | Cross-view attention + per-view MLP |
| Extra inputs | None | Confidence + world ray direction |
| Temperature | Implicit / fixed | Learnable `softplus(log_tau)` |
| Regularisation | Budget + min-views | Budget + min-views + sharpness |
| Output signature | `pred, weights, gate, reg` | `pred, weights, gate, reg` (same) |

## Testing and Validation

### Smoke test

`experiments/smoke_test_adaptive_softgate_v2.py` exercises:

1. **Shape and gradient test:** forward/backward on synthetic `(B=2, T=5, V=4, J=17)` data. Verifies output shapes and non-zero gradients.
2. **Gate validity test:** gate values lie in `[0, 1]` and show non-zero variance across views.
3. **Iterative refinement test:** `n_iter=3` forward pass completes without error.

Run it with:

```bash
python experiments/smoke_test_adaptive_softgate_v2.py
```

### Module self-test

The model file includes a `__main__` block that runs a forward/backward sanity check:

```bash
python -m motionflow_mv.fusion.ray_attention_temporal_residual_campe_adaptive_softgate_v2_model
```

### Test environment

- Python 3.10.20
- PyTorch 2.4.0+cu121
- CPU-only smoke run (GPU available but not required)

## Expected Impact

- **Robustness to occlusion.** By attending across views and seeing confidence/ray information, the gate should down-weight occluded or low-quality views more reliably than v1.
- **Better triangulation stability.** Cross-view competition should naturally favour geometrically strong views (wide baseline, clean rays) over redundant ones.
- **No hard selection.** The gate remains fully differentiable, so training is stable and no Gumbel/straight-through tricks are needed.
- **Drop-in replacement.** The v2 model has the same interface as v1 and can be trained with the same script by swapping the import.

## Limitations and Blockers

- **No real-data validation yet.** The smoke test only checks shapes, gradients, and gate validity. A short real-data training run is needed to confirm MPJPE.
- **v1 import bug not fixed.** v1 imports `_cameras_to_tensors` from `ray_attention_model`, but the function is named `cameras_to_tensors` there. v2 uses the correct import.
- **Camera positional encoding path unchanged.** We do not modify the CamPE module or the residual head, keeping the change minimal.
- **Sharpness regulariser is heuristic.** Its weight (0.1) may need tuning during full training.

## Next Steps

1. Add a training script `experiments/train_ray_attention_temporal_residual_campe_adaptive_softgate_v2_mpiinf3dhp.py` mirroring the v1 trainer.
2. Run 1–2 epoch smoke training on MPI-INF-3DHP to verify loss convergence and compare MPJPE against v1.
3. Evaluate robustness under synthetic occlusion (10%, 30%, 50%).
4. Tune `lambda_gate` and the sharpness coefficient if the gate collapses or stays too uniform.

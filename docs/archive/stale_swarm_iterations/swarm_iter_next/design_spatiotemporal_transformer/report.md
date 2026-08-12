# Cross-View Spatio-Temporal Transformer (task_07)

## Motivation

Existing ray-attention models factor the problem into separate stages:

* `RayAttentionFusionModelTemporal` attends over time for each `(view, joint)` pair.
* `RayAttentionFusionModelTemporalCrossview` attends over `(time, view)` tokens for each joint.

This leaves the **joint** dimension isolated: different body joints cannot directly exchange information during attention, even though biomechanical relationships (left/right symmetry, limb lengths, torso stability) are strong cues for 3D pose.  The goal of task_07 is therefore to build a single transformer that attends jointly over **time, views, and joints**, yielding a true cross-view spatio-temporal model.

## Design

### Model: `RayAttentionFusionModelSpatiotemporal`

File: `motionflow_mv/fusion/ray_attention_spatiotemporal_model.py`

The architecture reuses the v3 per-frame encoder and then stacks a unified transformer over the full `(T, V, J)` token grid.

```
Input: (B, T, V, J, 3)  -- (x, y, confidence)

1. Per-frame encoder (same as v3/temporal)
   - observation + ray embedding
   - camera-conditioned embedding
   - view-level self-attention
   - joint-level self-attention
   -> features f ∈ (B, T, V, J, d)

2. 3-D positional encoding
   - time embedding   e_t ∈ R^{T x d}
   - view embedding   e_v ∈ R^{V x d}
   - joint embedding  e_j ∈ R^{J x d}
   - f' = f + e_t + e_v + e_j

3. Spatio-temporal transformer
   - flatten (T, V, J) -> (B, T*V*J, d)
   - apply n_st_layers of standard TransformerEncoderLayer
   - reshape back to (B, T, V, J, d)

4. Weight head + differentiable weighted DLT triangulation
   -> raw 3D pose X_raw ∈ (B, T, J, 3)

5. Optional residual refinement head (default enabled)
   - pool per-view features per joint
   - MLP predicts ΔX from [pooled_feat, X_raw]
   -> final 3D pose X = X_raw + ΔX
```

### Key differences from prior models

| Model | Attention domain | Positional encodings | Residual head |
|---|---|---|---|
| `RayAttentionFusionModelTemporal` | time only | time | no |
| `RayAttentionFusionModelTemporalResidual` | time only | time | yes |
| `RayAttentionFusionModelTemporalCrossview` | time × view | time + view | no |
| `RayAttentionFusionModelTemporalCrossviewResidual` | time × view | time + view | yes |
| **RayAttentionFusionModelSpatiotemporal** | **time × view × joints** | **time + view + joint** | **yes** |

### Complexity notes

For a typical MPI-INF-3DHP clip (`T=13`, `V=4`, `J=17`), the flattened sequence has `884` tokens.  With `d=64` and `n_st_layers=2`, this is comparable in memory to the cross-view model (which uses `T*V=52` tokens per joint but keeps a joint-attention stage).  The unified model is therefore practical on standard GPUs.

## Files

* `motionflow_mv/fusion/ray_attention_spatiotemporal_model.py` — model implementation.
* `experiments/train_spatiotemporal_mpiinf3dhp.py` — training script for MPI-INF-3DHP clips.
* `tests/test_ray_attention_spatiotemporal.py` — forward/backward/shape sanity tests.
* `docs/swarm_iter_next/design_spatiotemporal_transformer/report.md` — this report.

## Validation

### Unit tests

```bash
conda run -n mf python -m pytest tests/test_ray_attention_spatiotemporal.py -v
```

Result: 3/3 tests passed.

### Smoke training

```bash
conda run -n mf python experiments/train_spatiotemporal_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --epochs 2 --batch_size 2 --train_samples 20 --d 32 --n_st_layers 1
```

Result:

```
Epoch 1: train_loss=0.009448, val_MPJPE=74.40mm (saved)
Epoch 2: train_loss=0.001268, val_MPJPE=33.29mm (saved)
Best val MPJPE: 33.29mm -> outputs/ray_attention_spatiotemporal_mpiinf3dhp.pth
```

The smoke run confirms that gradients flow through the full `(T, V, J)` attention grid, the DLT triangulation layer, and the optional residual head.

## Expected impact

1. **Stronger temporal consistency**: attending over the full token grid lets the model explicitly reason about joint-joint relationships across time, which should reduce flickering and implausible poses.
2. **Better occlusion handling**: when a joint is occluded in some views, information from the same joint in other views *and* from neighboring joints can propagate through a single attention operation.
3. **Minimal added parameter count**: the model reuses the existing per-frame encoder and only adds joint positional embeddings plus the shared transformer layers.

## Next steps / blockers

* **Full training**: the model has not been trained to convergence on MPI-INF-3DHP; only smoke tests have been run.  A full run on the canonical `_m` clips is needed to measure MPJPE against the 10.46 mm baseline.
* **Memory scaling**: for longer clips or higher-resolution skeletons, the `O((T*V*J)^2)` attention cost may become limiting.  If so, factorised axial attention (separate time/view/joint axes) is a natural next variant.
* **Positional encoding alternatives**: learned 3-D embeddings are a starting point; sinusoidal or camera-aware position encodings could be explored.
* **FusionModule wrapper**: to plug this model into the existing MotionFlow pipeline, a `RayAttentionSpatiotemporalFusionModule` wrapper analogous to `RayAttentionTemporalResidualFusionModule` should be added.

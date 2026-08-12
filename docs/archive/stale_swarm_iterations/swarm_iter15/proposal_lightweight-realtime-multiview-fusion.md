# Lightweight Real-Time Multi-View Fusion: Geometry-Aware Dynamic View Selection

## One-sentence hypothesis

Augmenting the 9.32 mm anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) with a closed-form epipolar-line-distance bias *and* a lightweight per-view/per-joint dynamic selection gate improves cross-view robustness and physical-space calibration while staying within real-time runtime constraints.

## Related existing files/modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — current 9.32 mm anchor.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_model.py` — anchor + epipolar bias on the weight head.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model.py` — anchor + dynamic view-selection gate.
- `motionflow_mv/fusion/epipolar_attention_bias.py` — differentiable epipolar-distance utilities.
- `motionflow_mv/fusion/dynamic_view_selection_gate.py` — soft per-view/per-joint gate.
- `motionflow_mv/losses/view_selection_loss.py` — sparsity + entropy regulariser for the gate.
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` — existing training harness that already supports `epipolar` and `dynamic_gate` variants.

## Proposed code changes

### 1. New model file

**File:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_dynamic_gate_model.py`

**Class:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarDynamicGate`

- Subclasses the anchor `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
- Adds a `DynamicViewSelectionGate(d=self.d, n_views=self.n_views)` after the spatio-temporal transformer.
- Computes per-joint epipolar-line distances from the *principal-point-corrected* intrinsics.
- Converts distances to an additive bias on the per-view weight logits via `epipolar_bias_from_distance(...)` with a learnable scalar blend (`epipolar_gate`).
- Multiplies the final triangulation weights by `gate_weights` in addition to confidences and visibility.
- Returns `(pred_3d, weights, gate_weights, gate_logits)` when `return_gate=True` and `return_pp_delta=False`.

**Signature additions (all optional, backward-compatible defaults):**

```python
def __init__(
    self,
    *args,
    epipolar_temperature: float = 100.0,
    gate_sparsity_weight: float = 0.01,   # consumed/ignored; trainer uses ViewSelectionLoss
    gate_entropy_weight: float = 0.001,   # consumed/ignored; trainer uses ViewSelectionLoss
    return_gate: bool = True,
    **kwargs,
):
```

### 2. Optional registration

- Add a matching `FusionModule` wrapper in `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_dynamic_gate_module.py` (mirrors the existing PP module).
- Register it in `motionflow_mv/fusion/__init__.py` so the pipeline can load it by name.

### 3. Optional training-script branch

- Add a new `--model_type epipolar_dynamic_gate` branch in `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` (or a new thin wrapper script), reusing the existing gate regulariser and reprojection losses.

## Training / smoke plan

1. **Smoke test:**
   ```bash
   python motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_dynamic_gate_model.py
   ```
   The file contains a CPU-only toy forward pass with `B=2, T=3, V=4, J=17` and should print shapes and exit cleanly.

2. **Short validation run (≤5 epochs):**
   ```bash
   python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
       --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
              data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
       --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
       --clip_len 9 --d 32 --residual_hidden 64 --n_st_layers 2 --epochs 5 \
       --batch_size 8 --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
       --gate_sparsity_weight 0.01 --gate_entropy_weight 0.001 \
       --reproj_refined_weight 0.05 --velocity_loss_weight 0.01
   ```
   *Estimated runtime on the local RTX 4090:* 15–25 min for  epochs on the two-sequence MPI-INF-3DHP subset.

3. **Cross-dataset sanity:** If the smoke run meets the MPJPE threshold, run the saved checkpoint through the existing H36M/Shelf/Campus evaluation on the read-only A800 for a one-number cross-dataset check.

## Success metrics

| Metric | Target |
|--------|--------|
| Clean MPJPE on MPI-INF-3DHP S2/Seq1 | < 9.0 mm (improvement over anchor 9.32 mm); regressions beyond 9.5 mm are considered a failure. |
| Robustness to view dropout / 2-D noise | Relative error increase under synthetic view dropout (rate 0.1–0.3) lower than the anchor. |
| Runtime | Forward latency within 10% of the anchor on RTX 4090 at `d=32, T=13, V=4, J=17`. |
| Gate sanity | Mean gate weight for synthetic occluded views < mean gate weight for clean views. |

## Risk and fallback

- **Risk:** The combination of epipolar bias and learned gate can overfit or destabilise training if the gate collapses to all-zeros/all-ones.
- **Fallback:** The change is deliberately modular.
  - If the dynamic gate hurts, set `gate_weights = 1.0` (one-line revert) and keep only the epipolar bias.
  - If the epipolar bias hurts, clamp `self.epipolar_gate` to a large negative value so its sigmoid is 0 (one-line revert).
  - If both hurt, delete the new file and continue with the original anchor.

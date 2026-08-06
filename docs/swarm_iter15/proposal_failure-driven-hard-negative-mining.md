# Failure-Driven Hard Negative Mining for Robust Multi-View 3D Pose

## One-sentence hypothesis

Online hard-negative mining that re-weights high-error joints and samples, plus a lightweight synthetic hard-negative generator that corrupts the most confident views, will push the anchor model past its 9.32 mm clean MPJPE ceiling by explicitly teaching the fusion network to recover from its own failure modes.

## Related existing files/modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` – anchor model (iter14 best, clean MPJPE 9.32 mm on MPI-INF-3DHP S2/Seq1).
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py` – parent model providing the spatio-temporal (time + view) transformer and residual refinement head.
- `motionflow_mv/losses/reprojection.py` – existing auxiliary reprojection loss; the new loss sits alongside it.
- `motionflow_mv/losses/view_selection_loss.py` – reference for per-sample/view regularisation patterns.
- `motionflow_mv/data/occlusion_aug.py` – existing occlusion utilities; reused for synthetic hard-negative generation.
- `motionflow_mv/calibration/perturb.py` – camera perturbation utilities; reused for calibration-aware hard negatives.
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` – current training script to be extended with a `--hard_negative_weight` flag.

## Proposed code changes

### 1. New loss module: `motionflow_mv/losses/hard_negative_mining_loss.py`

Implements two complementary mechanisms:

1. **Online Hard Example Mining (OHEM)**  
   Computes per-joint L2 error after each forward pass, selects the top-`k` hardest joints per batch, and applies a higher loss weight to those joints. The rest of the joints are still supervised with normal weight to avoid overfitting to outliers.

2. **Small memory bank of hard samples**  
   Maintains a fixed-size FIFO queue of the highest-error samples seen during an epoch. At regular intervals, the current batch is mixed with a sampled hard batch from the memory bank to prevent catastrophic forgetting of rare failure modes.

Public API:

```python
class HardNegativeMiningLoss(nn.Module):
    def __init__(self, base_loss='mse', ohem_ratio=0.25, hard_weight=2.0,
                 memory_size=256, memory_prob=0.0):
        ...

    def forward(self, pred, target, weights=None, return_mask=False):
        """
        pred:  (B, T, J, 3) or (B, J, 3)
        target: same shape as pred
        weights: optional (B, T, J) or (B, J) confidence mask
        returns: loss, (optional) hard-negative mask
        """
```

Signature change: none of the existing models need to change; the loss is consumed in the training script.

### 2. New model file: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_hard_negative_model.py`

Subclasses `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` and exposes a helper for synthetic hard-negative generation inside the model (kept in `forward` only when requested by a training hook). The model itself stays functionally identical at inference time.

Public API:

```python
class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointHardNegative(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    def __init__(self, ..., hard_negative_generator: Optional[nn.Module] = None):
        super().__init__(...)
        self.hard_negative_generator = hard_negative_generator or SyntheticHardNegativeGenerator()
```

No signature change is required in the training script beyond adding a `--hard_negative_weight` argument and passing it to the loss.

### 3. Synthetic hard-negative generator: `SyntheticHardNegativeGenerator`

A lightweight data-free module that operates on the input tensor `(B, T, V, J, 3)` and camera parameters:

- Compute per-view reprojection error of the raw triangulated pose.
- Identify the views with the smallest error (the "easy" views that the model trusts most).
- Corrupt those easy views with calibrated perturbations (small rotation/translation noise + 2-D outlier blobs) to synthesise plausible-but-wrong multi-view inputs.
- The same ground-truth 3D pose remains the target, forcing the network to distrust misleadingly consistent views.

This generator is only active during training and is fully deterministic given a fixed RNG seed.

## Training/smoke plan (≤5 epochs)

1. **Dataset:** MPI-INF-3DHP train = `s_01_seq_01_v14_multiview_m.npz` + `s_01_seq_02_v14_multiview_m.npz`; val = `s_02_seq_01_v14_multiview_m.npz`.
2. **Base command (single RTX 4090, ~2.5 h):**

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
    --epochs 5 --batch_size 8 --lr 1e-3 \
    --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
    --hard_negative_weight 1.5 --ohem_ratio 0.25 --hn_memory_size 256 \
    --output outputs/ray_attention_temporal_crossview_residual_principal_point_hn_mpiinf3dhp.pth
```

3. **Smoke test (≤1 epoch, ~20 min):**

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
    --epochs 1 --batch_size 8 \
    --hard_negative_weight 1.5 --ohem_ratio 0.25 \
    --output tmp/smoke_hn.pth
```

## Success metrics

| Metric | Target | How measured |
|--------|--------|--------------|
| Clean MPJPE on MPI-INF-3DHP S2/Seq1 | < 9.32 mm (anchor) | `experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` |
| View-dropout robustness | ≤ 5 % MPJPE increase at 20 % view dropout | same eval with `--view_dropout_rate 0.2` |
| Calibration robustness | ≤ 3 % MPJPE increase at `cam_aug_pp 10.0` | eval with perturbed intrinsics |
| Convergence stability | no NaN/inf losses in ≤5 epochs | training log |

## Risk and fallback

- **Risk:** OHEM can overfit to annotation outliers or motion blur, causing clean-set MPJPE to degrade.  
  **Fallback:** Drop the memory bank and use a simple top-`k` per-joint reweighting with a smaller `hard_weight` (e.g., 1.25). This is a single-line hyperparameter change.

- **Risk:** Synthetic hard negatives made by corrupting the best views may create physically implausible camera configurations, destabilising the principal-point correction head.  
  **Fallback:** Disable the synthetic generator and keep only the OHEM loss, which has no extra data requirements and is guaranteed not to break the camera model.

- **Risk:** Training runtime increases because of the extra forward pass for reprojection error.  
  **Fallback:** Compute the hard-negative mask from the already-available 3D prediction error, avoiding any extra forward pass.

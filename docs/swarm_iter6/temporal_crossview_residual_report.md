# Temporal + Cross-View Residual Refinement Model

**Date:** 2026-08-04
**Author:** Kimi Code sub-agent
**Goal:** Add cross-view attention to the top-performing residual refinement head and smoke-train on MPI-INF-3DHP to see if residual + cross-view beats residual alone.

## What was implemented

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
  - New `RayAttentionFusionModelTemporalCrossviewResidual` model.
  - Extends `RayAttentionFusionModelTemporalCrossview` (the spatio-temporal time + cross-view transformer) by adding the same residual MLP head used in `RayAttentionFusionModelTemporalResidual`.
  - After the time + cross-view transformer and weighted DLT triangulation, the model pools the per-view spatio-temporal features per joint, concatenates them with the raw triangulated 3D pose, and predicts a per-joint residual correction ΔX that is added back to the raw estimate.
  - Keeps the same lightweight residual MLP (two hidden layers of size `residual_hidden`, default 128).
  - Adds only the residual head parameters to the cross-view temporal model, keeping the change minimal and focused.

- `experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py`
  - Mirrors the residual training script but loads `RayAttentionFusionModelTemporalCrossviewResidual`.
  - Uses `n_st_layers` (default 2) instead of `n_temporal_layers` to match the cross-view temporal model's constructor.
  - Defaults to `outputs/ray_attention_temporal_crossview_residual_mpiinf3dhp.pth`.

## Sanity check

A forward/backward shape sanity check passed:

```bash
conda run -n mf python - <<'PY'
import torch
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidual, _make_cameras,
)
B, T, V, J = 2, 5, 4, 17
x = torch.rand(B, T, V, J, 3)
cameras = _make_cameras(V)
model = RayAttentionFusionModelTemporalCrossviewResidual(j=J, d=64, n_views=V)
pred, w = model(x, cameras=cameras)
print("pred shape:", pred.shape)      # (B, T, J, 3)
print("weights shape:", w.shape)      # (B, T, V, J)
loss = pred.mean()
loss.backward()
print("crossview residual sanity check passed")
PY
```

Output: `pred shape: torch.Size([2, 5, 17, 3])`, `weights shape: torch.Size([2, 5, 4, 17])`, and gradients flow correctly.

## Smoke-test comparison (smoke .npz files)

Both models were trained on the same smoke sequences with identical hyperparameters (10 epochs, clip_len=13, batch_size=4, train_samples=50):

```bash
conda run -n mf python experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --batch_size 4 --train_samples 50 --epochs 10 \
    --output outputs/ray_attention_temporal_crossview_residual_smoke10.pth
```

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --batch_size 4 --train_samples 50 --epochs 10 \
    --output outputs/ray_attention_temporal_residual_smoke10.pth
```

### Results

| Model | Params | Best val MPJPE | 10-epoch trajectory (mm) |
|---|---|---|---|
| Temporal residual (baseline) | 243,428 | **17.01 mm** | 19.03 → 18.49 → 18.28 → 18.18 → 20.56 → 18.89 → 21.87 → 19.49 → 23.96 → 17.01 |
| Temporal + cross-view residual | 244,324 | **11.56 mm** | 22.70 → 22.76 → 11.56 → 19.47 → 16.44 → 11.99 → 21.41 → 17.43 → 16.41 → 14.84 |

The cross-view residual variant beat the residual-only baseline by **5.45 mm** on the smoke validation set (S2 Seq1 smoke), suggesting that the time + cross-view transformer provides useful complementary information to the residual head.

## Implementation notes

- **No new dependencies:** The implementation reuses existing PyTorch modules and project utilities.
- **Memory:** The spatio-temporal transformer attends over `T * V` tokens per joint. With `T=13` and `V=14` this yields 182 tokens per joint; memory stayed within the RTX 4090's 24 GB for batch size 4.
- **Parameter count:** The cross-view residual model has only ~900 more parameters than the temporal residual model, all from the time + view positional embeddings and the extra dimension handled by the spatio-temporal transformer.

## Blockers / follow-up

- The smoke files are short (250 frames) and the improvement, while encouraging, may not generalize. A longer run on the full MPI-INF-3DHP validation set (S2 Seq1) is needed to confirm the gain.
- The cross-view transformer is slower per step than the temporal-only transformer; a full run should budget more wall-clock time or use a smaller batch size.
- Future work: evaluate the trained model on H36M / 3DPW to test cross-dataset generalization.

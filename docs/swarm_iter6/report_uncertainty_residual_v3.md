# Uncertainty-aware residual refinement (v3)

**Goal:** Add uncertainty estimation to the top-performing residual refinement head:
* the weight head should predict both per-view weights and per-joint uncertainties, and
* the residual head should use uncertainty-aware features.

## Files added / modified

| Path | Purpose |
|------|---------|
| `motionflow_mv/fusion/ray_attention_temporal_residual_v3_model.py` | New model `RayAttentionFusionModelTemporalResidualV3` extending the residual model with an 2-channel weight/uncertainty head and an uncertainty-aware residual MLP. |
| `experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py` | New training script; reuses dataset/augmentation/evaluation helpers from `train_ray_attention_temporal_residual_mpiinf3dhp.py` and instantiates the v3 model. |
| `tests/test_ray_attention_temporal_residual_v3.py` | Shape, single-frame, and finite-ness sanity tests for the v3 model. |

## Architecture changes

`RayAttentionFusionModelTemporalResidualV3` subclasses `RayAttentionFusionModelTemporalResidual` and keeps the same temporal ray-attention backbone.

1. **Weight/uncertainty head**
   ```python
   self.weight_head = nn.Linear(self.d, 2)  # (weight_logit, uncertainty_logit)
   ```
   * `weight = sigmoid(logit)` is multiplied by the input confidence and used in weighted DLT triangulation, exactly as before.
   * `uncertainty = softplus(logit) + eps` is a strictly positive per-view per-joint uncertainty.

2. **Uncertainty-aware residual head**
   * Per-view temporal features `feat` (after temporal transformer) are pooled across views with inverse-uncertainty weights:
     ```python
     inv_u = 1.0 / uncertainty
     u_weights = inv_u / inv_u.sum(dim=1, keepdim=True)
     feat_pooled = (feat * u_weights[..., None]).sum(dim=1)  # (B*T, J, d)
     ```
   * A per-joint log-uncertainty summary is concatenated:
     ```python
     log_unc_summary = torch.log(uncertainty).mean(dim=1)  # (B*T, J)
     ```
   * The residual MLP now consumes `d + 3 + 1` inputs:
     ```python
     residual_input = concat(feat_pooled, pred_3d_raw, log_unc_summary)
     delta = residual_mlp(residual_input)
     pred_3d = pred_3d_raw + delta
     ```

This keeps the change minimal and directly satisfies the two requirements: the weight head predicts both weights and uncertainty, and the residual head explicitly reasons about the predicted uncertainty.

## Sanity checks

```bash
conda run -n mf python tests/test_ray_attention_temporal_residual_v3.py
# temporal residual v3 uncertainty tests passed
```

## Smoke training

Because the swarm GPU (RTX 4090) is shared with many agents, the smoke run was kept short:

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 9 --d 16 --n_temporal_layers 1 --residual_hidden 32 \
    --batch_size 2 --train_samples 200 --epochs 2 \
    --output outputs/ray_attention_temporal_residual_v3_mpiinf3dhp_smoke.pth
```

Output:

```
Device: cuda
n_views=14, j=28, clip_len=9, d=16, residual_hidden=32
Train clips: 200, Val clips: 242
Model params: 17164
Epoch 1: train_loss=0.001478, val_MPJPE=9.72mm, lr=1.00e-03 (saved)
Epoch 2: train_loss=0.000096, val_MPJPE=12.07mm, lr=1.00e-03
Best val MPJPE: 9.72mm -> outputs\ray_attention_temporal_residual_v3_mpiinf3dhp_smoke.pth
```

The model trains without errors and the checkpoint is saved. The smoke-scale validation (MPI-INF-3DHP S2 Seq1 smoke subset) reports a best MPJPE of **9.72 mm**. This is not directly comparable to the full-data ~13.84 mm baseline because the smoke subset is tiny and the run was only 2 epochs with a smaller model, but it confirms the new uncertainty branch is functional and learnable.

## Blockers / next steps

* **No blockers.** Forward/backward passes, training loop, and checkpoint saving all work.
* A full run with the original hyperparameters (`d=64`, `n_temporal_layers=2`, `residual_hidden=128`, 30 epochs, full MPI-INF-3DHP train/val) is needed to measure whether the uncertainty branch improves the ~13.84 mm full-data result.
* Optional future work:
  * Supervise the uncertainty head with a reprojection NLL or residual error to give it a stronger learning signal (the current model learns uncertainty implicitly as a free feature).
  * Experiment with uncertainty gating inside the residual MLP instead of simple concatenation.

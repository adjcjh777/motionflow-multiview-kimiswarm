# Proposal: Attention-Entropy Regularisation for Interpretable Multi-View Fusion

**Direction:** `attention_visualization_interpretability`  
**Related issues:** #23, #25  
**Date:** 2026-08-06  
**Author:** Kimi Code agent (swarm direction #19)  

## Motivation

The current best hierarchical attention model (`RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint`) reaches **~10.23 mm** on MPI-INF-3DHP, still above the 8.75 mm anchor.  Earlier interpretability work ([`docs/swarm_iter7/interpretability_failure_analysis.md`](../swarm_iter7/interpretability_failure_analysis.md)) showed that the model's per-view fusion weights are **positively** correlated with reprojection error, i.e. the model does not reliably down-weight noisy or occluded views.  This suggests that sharper, more interpretable view-selection could improve both accuracy and robustness.

This proposal adds an **attention-entropy regularisation** term that encourages the model to concentrate per-joint weight mass on a small subset of views.  A low-entropy weight distribution is:

1. **More interpretable** — it is easy to visualise which views are trusted for each joint.
2. **More robust** — by penalising diffuse attention, the model is forced to make crisper reliability decisions and ignore low-quality views.
3. **Minimal** — it re-uses the existing hierarchical backbone and only adds a scalar loss.

## Method

### New model variant

`motionflow_mv/fusion/ray_attention_hierarchical_attention_entropy_reg_model.py`

- Subclasses `RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint`.
- Computes the per-joint Shannon entropy of the normalised per-view triangulation weights.
- Returns the entropy as an auxiliary loss term.

Entropy of normalised weights `p_v = w_v / Σ w`:

```
H = - Σ_v p_v log(p_v)
```

The training loss becomes:

```
L = MSE(pred, gt) + λ_entropy * H
```

where `λ_entropy` is controlled by `--attention_entropy_weight`.

### Training script

`experiments/train_hierarchical_attention_entropy_reg_mpiinf3dhp.py`

- Dedicated training script copied from the principal-point trainer and stripped to only support the entropy-reg model.
- New CLI arg `--attention_entropy_weight` (default `0.01`).
- Adds the returned `entropy_loss` to the total MSE loss.

### Smoke test

- `tests/test_ray_attention_hierarchical_attention_entropy_reg.py`: unit tests for forward shapes, gradients, single-frame compatibility, and entropy monotonicity.
- `scripts/run_hierarchical_attention_entropy_reg_smoke_wsl.sh`: 2-epoch CPU smoke on the MPI-INF-3DHP smoke subset.

## Expected outcome

- A reproducible CPU smoke test that completes in <2 minutes.
- The entropy term should decrease as view weights become more concentrated (validated by unit test).
- A plausible path toward the 8.75 mm anchor: sharper view selection should reduce the influence of noisy cameras and improve cross-view robustness, especially under occlusion or calibration perturbation.

## How to run

### CPU smoke test

```bash
./scripts/run_hierarchical_attention_entropy_reg_smoke_wsl.sh
```

Log: `outputs/hierarchical_attention_entropy_reg_smoke.log`

### Unit tests

```bash
python -m pytest tests/test_ray_attention_hierarchical_attention_entropy_reg.py -v
```

### Full MPI-INF-3DHP run

```bash
python -u experiments/train_hierarchical_attention_entropy_reg_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 64 --residual_hidden 128 \
  --n_view_groups 2 --n_view_layers 2 --n_temporal_layers 2 --n_joint_graph_layers 1 \
  --epochs 20 --train_samples 1000 --batch_size 8 --val_stride 50 \
  --pp_loss_weight 0.2 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
  --pp_pretrain_epochs 3 --attention_entropy_weight 0.01 \
  --output outputs/hierarchical_attention_entropy_reg_mpiinf3dhp.pth
```

## Next validation step

Run the full 20-epoch MPI-INF-3DHP training above and compare MPJPE against the hierarchical attention baseline (`outputs/hierarchical_attention_pp_full_mpiinf3dhp.pth`).  If the entropy term helps, sweep `λ_entropy ∈ {0.001, 0.01, 0.05}` and check the robustness matrix under camera perturbation.

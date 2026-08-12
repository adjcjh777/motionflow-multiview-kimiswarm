# Uncertainty-aware per-view weighting

I investigated the topic and found that uncertainty-aware per-view weighting already exists in several forms, but the latest full run did not beat the current best residual baseline. I cannot write files in this read-only exploration mode, so the report content is below at the intended path.

**Intended path:** `docs/swarm_iter7/uncertainty-aware_per-view_weighting.md`

---

```markdown
# Uncertainty-aware per-view weighting

## 1. Current state

The project already has several uncertainty-aware or per-view-weighting components, but none have beaten the current best residual baseline.

Relevant existing code:

- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py` — current best model `RayAttentionFusionModelTemporalResidual` (11.17 mm MPJPE on MPI-INF-3DHP S1→S2/Seq1). It predicts per-view per-joint sigmoid weights and multiplies them by the input confidence before weighted DLT triangulation.
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_model.py` — stand-alone uncertainty-weighted triangulation model. It replaces the sigmoid weight head with a Gaussian log-variance head and uses `weight = confidence * exp(-log_var)`. It also computes an auxiliary reprojection NLL loss.
- `motionflow_mv/fusion/ray_attention_temporal_residual_v3_model.py` — uncertainty-aware residual model (`RayAttentionFusionModelTemporalResidualUncertainty`). The weight head now predicts both a per-view weight and a per-view uncertainty; the residual MLP is fed with inverse-uncertainty-weighted features plus a per-joint log-uncertainty summary.
- `experiments/train_ray_attention_temporal_residual_uncertainty_mpiinf3dhp.py` and `experiments/eval_ray_attention_temporal_residual_uncertainty.py` — training/evaluation scripts for the uncertainty-aware residual model.
- `docs/swarm_iter6/report_uncertainty_residual_v3.md` and `docs/swarm_iter7/plan_uncertainty_residual.md` — prior swarm reports on the same topic.

Latest full run (from `outputs/train_residual_uncertainty_full5.log` and `outputs/eval_uncertainty_epoch2.json`):

```text
Epoch 1: val_MPJPE=14.80mm
Epoch 2: val_MPJPE=12.89mm  (best)
Epoch 3: val_MPJPE=14.31mm
```

Final evaluation of the best checkpoint: **MPJPE 12.89 mm**, **PA-MPJPE 11.62 mm**, **AUC 0.9141**. This is worse than the current best 11.17 mm MPJPE / 8.24 mm PA-MPJPE from the plain residual model.

## 2. Gap / opportunity

The existing uncertainty-aware residual model learns uncertainty **implicitly**: the uncertainty branch is trained only through the MSE loss on the final 3D pose. There is **no explicit supervision** tying the predicted uncertainty to the actual per-view reprojection error. Consequently, the uncertainty head may not form a reliable per-view confidence estimate, and the residual head receives a weak/noisy conditioning signal.

The stand-alone `ray_attention_temporal_uncertainty_model.py` already contains a proper Gaussian reprojection NLL loss (`_reprojection_nll`), but it was never integrated into the residual-uncertainty model's training loop. Adding explicit reprojection-NLL supervision is the most direct, under-explored opportunity to make the per-view weighting actually uncertainty-aware.

## 3. Concrete next step

Add the explicit reprojection-NLL auxiliary loss to the uncertainty-aware residual model and run a full training/evaluation cycle.

Suggested implementation:

1. In `motionflow_mv/fusion/ray_attention_temporal_residual_v3_model.py`, expose the predicted `log_var` (or uncertainty) alongside the existing `weights` and `pred_3d`.
2. Port the `_reprojection_nll` helper from `ray_attention_temporal_uncertainty_model.py` into the residual v3 model (or share a utility in `motionflow_mv/fusion/triangulation.py`).
3. In `experiments/train_ray_attention_temporal_residual_uncertainty_mpiinf3dhp.py`, change the training loss from:
   ```python
   loss = criterion(pred, yb)
   ```
   to:
   ```python
   pred, weights, log_var, nll_loss = model(xb, K=K, R=R, t=t)
   loss = criterion(pred, yb) + 0.1 * nll_loss
   ```
   (or make the NLL weight a command-line argument).
4. Run a full MPI-INF-3DHP experiment mirroring the current best residual hyperparameters:
   ```bash
   conda run -n mf python experiments/train_ray_attention_temporal_residual_uncertainty_mpiinf3dhp.py \
       --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
               data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
       --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
       --clip_len 13 --d 64 --residual_hidden 128 --n_temporal_layers 2 \
       --epochs 5 --batch_size 8 --train_samples 4000 \
       --output outputs/ray_attention_temporal_residual_uncertainty_nll_full5.pth
   ```
5. Evaluate the checkpoint with `experiments/eval_ray_attention_temporal_residual_uncertainty.py` and compare against the current best.

A lighter first step is to run a smoke test for 2 epochs with the NLL loss to verify the loss does not explode and the uncertainty values are reasonable.

## 4. Expected success metric

- Primary: **MPI-INF-3DHP S1→S2/Seq1 MPJPE < 11.17 mm** and/or **PA-MPJPE < 8.24 mm**, beating the current best residual model.
- Secondary: robustness improvement under 2D outliers/occlusion (the explicit uncertainty should down-weight corrupted views more reliably).
- Diagnostic: predicted per-view uncertainty should correlate with actual per-view reprojection error; a quick histogram of `log_var` vs. reprojection error should show a clear trend.

## 5. Risks / blockers

- **A800-D and Docker are read-only** — training must continue on the local RTX 4090.
- **NLL loss can dominate early training** (observed in the stand-alone uncertainty model, where the first epoch loss was ~24.5). Start with a small weight (e.g., 0.01–0.1) and normalize the reprojection error by image size if needed.
- **No large-file commits** — checkpoints and logs stay in `outputs/`; do not commit them.
- **WebBridge data may need re-download** if the local `.npz` files are missing; do not commit downloaded datasets.
- **Class-name collision** was previously resolved by renaming the uncertainty class to `RayAttentionFusionModelTemporalResidualUncertainty`; verify imports before launching.
```
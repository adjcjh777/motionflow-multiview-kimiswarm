# Uncertainty-Weighted Triangulation (v2) — Implementation Report

## What was implemented
A new model, `RayAttentionFusionModelTemporalUncertaintyV2`, and a matching
training script, `experiments/train_uncertainty_v2_mpiinf3dhp.py`, that adds a
learned uncertainty head to the temporal residual model.

## Files created
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py`
- `experiments/train_uncertainty_v2_mpiinf3dhp.py`
- `docs/swarm_iter_next/design_uncertainty_weighted_triangulation/plan.md`
- `docs/swarm_iter_next/design_uncertainty_weighted_triangulation/report.md`

## Architecture highlights
- Inherits from `RayAttentionFusionModelTemporalResidualV2` to reuse:
  - V4-normalised camera embedding.
  - Temporal view/joint attention pipeline.
  - Residual refinement MLP and optional reprojection-error gate.
- Adds `uncertainty_head`: a single linear layer on the per-view/joint temporal
  feature token that predicts `log_var`.
- DLT weights are computed as `precision * confidence = exp(-log_var) * confidence`.
- Reprojection NLL loss encourages the predicted uncertainties to match actual
  reprojection errors.

## How to test / validate
Model sanity check (CPU/small synthetic data):
```bash
conda run -n mf python -m motionflow_mv.fusion.ray_attention_temporal_uncertainty_v2_model
```
Expected output:
```
temporal uncertainty v2 model sanity check passed
iterative refinement sanity check passed
```

One-epoch smoke training (uses H36M because no MPI-INF-3DHP .npz is present in
the current checkout):
```bash
conda run -n mf python experiments/train_uncertainty_v2_mpiinf3dhp.py \
    --train data/h36m_hf/s_01_train_subset_500_m.npz \
    --val data/h36m_hf/s_01_train_subset_500_m.npz \
    --clip_len 9 --batch_size 2 --train_samples 20 --epochs 1 \
    --output outputs/ray_attention_temporal_uncertainty_v2_smoke.pth
```
The smoke run completed successfully on this environment:
- Device: cuda
- Model params: 202,021
- Epoch 1: train_loss=39.839147, val_MPJPE=121.96 mm (saved)

## Expected impact
- **Robustness**: down-weighting noisy/occluded views via learned precision should
  reduce the influence of bad 2D detections during triangulation.
- **Interpretability**: per-view log-variance provides a direct measure of the
  model’s confidence in each camera view.
- **Compatibility**: the v2 camera embedding preserves the ability to warm-start
  from V4 checkpoints and to share encoders with other v2-family models.

## Blockers / next steps
- Full MPI-INF-3DHP training data is required for a meaningful metric comparison
  against the 10.46 mm baseline. The current checkout does not contain the
  `data/webbridge/mpi_inf_3dhp/*.npz` files.
- The default `uncertainty_weight=0.1` should be tuned; higher values can
  destabilise early training.
- A dedicated evaluation script (mirroring
  `eval_ray_attention_temporal_residual_mpiinf3dhp.py`) has not been created;
  the existing eval scripts can be adapted once a trained checkpoint is
  available.

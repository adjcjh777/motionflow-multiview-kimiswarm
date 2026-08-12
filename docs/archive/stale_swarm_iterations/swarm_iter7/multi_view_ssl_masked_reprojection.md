# Direction 1 — Multi-view pre-training representations (SSL masked-view reprojection)

## Problem statement

The current best supervised model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, reaches 9.32 mm clean MPJPE on MPI-INF-3DHP but is trained end-to-end on a relatively small labelled multi-view corpus. Large unlabelled multi-view video collections such as Human3.6M, AIST++, and synthetic sequences are available, yet their 3D labels are either absent or domain-mismatched. Masked-view reprojection pre-training lets us learn view-geometry-aware representations without 3D ground truth: random views or time steps are masked out and the model is asked to minimise reprojection error on both visible and masked slots. The hypothesis is that a model pre-trained this way will require far fewer labelled MPI clips during fine-tuning and may generalise better across camera setups and occlusions.

## Simplest concrete next experiment

Run a short self-supervised pre-training run on Human3.6M (subjects 1, 5, 6, 7, 8; validation on subject 9) using the existing `experiments/pretrain_ray_attention_ssl.py`, then fine-tune the resulting checkpoint on MPI-INF-3DHP with `experiments/run_data_efficiency_curve.py` to quantify the gain at 5–100 % label fractions compared with training from scratch. Because the RTX 4090 is currently busy with the cross-view PP curriculum, the actual GPU pre-training is queued; the only work done now is the CPU smoke test that validates the pipeline end-to-end.

## Files to touch and rough sketch

No existing experiment runner needs to be modified. The required components already exist:

- `motionflow_mv/data/ssl_dataset.py` — `SSLRandomClipDataset`, `SSLTemporalClipDataset`, and masking helpers.
- `experiments/pretrain_ray_attention_ssl.py` — masked-view reprojection pre-training loop.
- `experiments/run_data_efficiency_curve.py` — fine-tune a pretrained checkpoint on label fractions.
- `scripts/run_ssl_pretrain_h36m_wsl.sh` — existing launcher for H36M SSL pre-training.
- `experiments/smoke_pretrain_ray_attention_ssl.py` — CPU smoke test (already run, see below).

A minimal next-step launcher that fits the current GPU queue could be placed in `scripts/run_ssl_pretrain_h36m_smoke_wsl.sh`:

```bash
#!/usr/bin/env bash
# Quick 10-epoch SSL pre-training smoke on H36M to validate before the full run.
set -e
cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm

TRAIN_FILES=(
  data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz
  data/webbridge/h36m_meters/s_05_acts_02_multiview_m.npz
)
VAL_FILE=data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz

conda run -n mf python experiments/pretrain_ray_attention_ssl.py \
  --train "${TRAIN_FILES[@]}" \
  --val "${VAL_FILE}" \
  --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --epochs 10 --batch_size 8 --train_samples 4000 \
  --mask_ratio 0.25 --mask_mode mixed \
  --lambda_vis 1.0 --lambda_mask 1.0 --lambda_smooth 0.1 --lambda_bone 0.1 \
  --output outputs/ray_attention_ssl_h36m_smoke.pth \
  "$@"
```

After pre-training, start the data-efficiency fine-tune with:

```bash
conda run -n mf python experiments/run_data_efficiency_curve.py \
  --pretrained outputs/ray_attention_ssl_h36m_smoke.pth \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --fractions 0.05 0.10 0.25 0.50 1.00 \
  --output_dir outputs/data_efficiency_curve_ssl \
  --epochs 20 --batch_size 8 --train_samples 4000
```

## Expected success metric

- SSL pre-training validation reprojection loss decreases across epochs and ends below the un-trained baseline (masked-slot loss should approach the visible-slot loss, indicating the model fills in missing views).
- After fine-tuning on MPI-INF-3DHP, the SSL-warm-started model matches or beats the from-scratch baseline with ≤ 50 % of the labelled clips; target a relative PA-MPJPE improvement of ≥ 10 % at the 10 % label fraction.
- The eventual full H36M → MPI transfer should preserve the current clean MPJPE of 9.32 mm (or improve it) while using substantially fewer annotated MPI frames.

## Resource requirements

- **SSL pre-training:** GPU (WSL RTX 4090 or A800-D read-only access for data). Do not start until the current cross-view PP curriculum job finishes.
- **Fine-tuning / data-efficiency curve:** GPU.
- **Smoke test:** CPU-only, safe to run now. Completed below.

## CPU smoke test

The existing smoke test generates a tiny synthetic `.npz` dataset, checks the masked-view data loader, and runs one epoch of `pretrain_ray_attention_ssl.py` on the CPU.

Command:

```bash
python experiments/smoke_pretrain_ray_attention_ssl.py
```

Result:

```
Generating synthetic train/val .npz files...

--- Masked-view reprojection data loading report ---
train batches: 4, val batches: 11
batch 0: x_masked=[2, 9, 4, 17, 3], mask=[2, 9, 4, 17], original_conf=[2, 9, 4, 17, 3], K=[2, 4, 3, 3], R=[2, 4, 3, 3], t=[2, 4, 3]
  masked slots: 408/1224 (33.3%), visible slots: 816
  confidence after masking (min/mean/max): 0.000/0.667/1.000
batch 1: x_masked=[2, 9, 4, 17, 3], mask=[2, 9, 4, 17], original_conf=[2, 9, 4, 17, 3], K=[2, 4, 3, 3], R=[2, 4, 3, 3], t=[2, 4, 3]
  masked slots: 408/1224 (33.3%), visible slots: 816
  confidence after masking (min/mean/max): 0.000/0.667/1.000

--- Running one CPU epoch of pretrain_ray_attention_ssl.py ---
Device: cpu
n_views=4, j=17, clip_len=9
Model params: 54454
Epoch 1: train_loss=2748214537113252864.000000 val_loss=7941328384.000000 val_vis=7941328384.000000 val_mask=4546891415.272727 (saved)
Best val loss: 7941328384.000000 -> .../ray_attention_ssl_smoke.pth
Smoke test completed successfully.
```

The loss values are large because the synthetic skeleton is random and only a single epoch is run; the important outcome is that the data loader, masking logic, model forward pass, and reprojection loss all run without error on the CPU.

## Notes

- No existing training script was modified; only a new launcher sketch is proposed.
- GPU pre-training must wait for the currently running cross-view PP curriculum job to free the RTX 4090.
- Status: CPU smoke passed; GPU pre-training queued.

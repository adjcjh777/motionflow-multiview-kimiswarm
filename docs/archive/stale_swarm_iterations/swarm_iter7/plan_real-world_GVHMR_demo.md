# Plan: Fine-tune residual ray-attention fusion for the real-world GVHMR demo

## What to change

1. **Extend `experiments/demo_gvhmr_multiview_projection.py`** to load the current best residual model:
   - Add `--ray_residual_checkpoint` to instantiate `RayAttentionFusionModelTemporalResidual`.
   - Keep the existing `--ray_v3_checkpoint` path for comparison.

2. **Add a GVHMR-like noise model to the demo** so the real GVHMR output is fused under realistic per-view errors:
   - `--gvhmr_noise_std` (default 2.0 px)
   - `--gvhmr_outlier_rate` (default 0.05)
   - `--gvhmr_dropout_rate` (default 0.10)

3. **Extend `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`** to accept augmentation and warm-start args:
   - `--noise_std`, `--dropout_rate`, `--outlier_rate`
   - `--pretrained outputs/ray_attention_temporal_residual_v2.pth`
   - Use a lower fine-tuning LR (e.g., 1e-4).

4. **Fine-tune the 11.17 mm MPI checkpoint on MPI-INF-3DHP with GVHMR-calibrated augmentation** to close the synthetic-to-real gap.

5. **Run the updated demo on the real `data/gvhmr_demo/hmr4d_results.pt`** and compare fused multi-view output against the single-view GVHMR world reference.

## Why it should help

- The current best residual model is trained on clean dataset keypoints and has never seen monocular SMPL-style errors (fitting bias, jitter, outliers). Injecting realistic GVHMR noise during fine-tuning makes the residual head robust to real-world detector errors without changing the architecture.
- Using the temporal-residual model in the demo replaces the older v3 path with the architecture that actually holds the 11.17 mm MPI record, raising the demo’s accuracy ceiling and removing the inconsistency between the paper’s best numbers and the demo code.
- It directly addresses the “no real-world GVHMR output evaluation” limitation in the CVPR/ICRA draft.

## Exact commands

### Smoke test (1–3 epochs, local RTX 4090)

```bash
# 1. Fine-tune from the current best checkpoint with GVHMR-style augmentation
conda run -n mf python experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --residual_hidden 128 \
    --epochs 3 --batch_size 4 --train_samples 1000 \
    --noise_std 2.0 --dropout_rate 0.1 --outlier_rate 0.05 \
    --pretrained outputs/ray_attention_temporal_residual_v2.pth \
    --output outputs/ray_attention_temporal_residual_gvhmr_smoke.pth

# 2. Run the real GVHMR demo with the fine-tuned residual model
conda run -n mf python experiments/demo_gvhmr_multiview_projection.py \
    --input data/gvhmr_demo/hmr4d_results.pt \
    --n_views 4 --noise_std 0.0 \
    --gvhmr_noise_std 2.0 --gvhmr_outlier_rate 0.05 \
    --ray_residual_checkpoint outputs/ray_attention_temporal_residual_gvhmr_smoke.pth \
    --max_frames 100

# 3. Sanity-check clean MPI accuracy is preserved
conda run -n mf python experiments/eval_ray_attention_temporal_residual_mpiinf3dhp.py \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_residual_gvhmr_smoke.pth \
    --clip_len 13 --d 64 --batch_size 8
```

### Full run

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --residual_hidden 128 \
    --epochs 20 --batch_size 8 --train_samples 4000 \
    --noise_std 2.0 --dropout_rate 0.1 --outlier_rate 0.05 \
    --pretrained outputs/ray_attention_temporal_residual_v2.pth \
    --output outputs/ray_attention_temporal_residual_gvhmr.pth

conda run -n mf python experiments/demo_gvhmr_multiview_projection.py \
    --input data/gvhmr_demo/hmr4d_results.pt \
    --n_views 4 --noise_std 0.0 \
    --gvhmr_noise_std 2.0 --gvhmr_outlier_rate 0.05 \
    --ray_residual_checkpoint outputs/ray_attention_temporal_residual_gvhmr.pth
```

## Expected metrics

- **GVHMR demo MPJPE vs single-view GVHMR world reference**:
  - Plain DLT baseline: ~0.040–0.050 m
  - `RayAttentionTemporalResidual` clean checkpoint: ~0.010–0.015 m
  - Fine-tuned GVHMR-style residual: **≤0.010 m** (target ≤0.008 m)

- **MPI-INF-3DHP S2/Seq1 clean validation**: keep MPJPE within **11.0–11.7 mm** to ensure no catastrophic forgetting.

- **PA-MPJPE / PCK@50/100/150** on MPI should stay within ±5% relative of the original checkpoint.

## Risks and mitigations

- **Missing real GVHMR demo file**: `data/gvhmr_demo/hmr4d_results.pt` is currently empty. Copy a real `.pt` from the A800-D GVHMR outputs before running.
- **No 3D GT for the demo**: evaluation is relative to the single-view GVHMR world output. Also report reprojection error on the virtual cameras for an absolute geometric check.
- **Catastrophic forgetting**: use low fine-tuning LR and keep the checkpoint with the lowest MPI validation MPJPE.
- **Per-view coordinate alignment**: this experiment uses one real GVHMR output + virtual views; a later per-view GVHMR inference demo must align each view’s gravity-view frame to the calibrated rig.
- **Windows NumPy BLAS instability**: run training/evaluation in WSL or on A800-D if the local Anaconda environment fails on linear algebra calls.

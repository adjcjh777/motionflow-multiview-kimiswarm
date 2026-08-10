# v80 (view-reliability weighting) on the H36M true-GT standard protocol

> **Date:** 2026-08-10 / 2026-08-11
> **Status:** First learned-model convergence attempt on the repaired H36M
> protocol (issue #194). All runs on A800-D GPUs 4,5, pinned via
> CUDA_VISIBLE_DEVICES, nohup; ≤2 GPU rule respected throughout.
> **Manifest:** `configs/splits/h36m_true_gt_standard.yaml` (S1,5,6,7,8 train,
> ~390k frames; S9/S11 val). **DLT anchor:** S9 29.54 / S11 21.81 mm MPJPE
> (combined mean 25.67; `data/h36m_true_gt/dlt_baseline_h36m.json`).
> **Iskakov-style learned-weight DLT anchor on the same labels:** combined
> direct 23.38 mm (`docs/results_iskakov_h36m_true_gt.md`).

## Recipe sweep

All recipes share the v80 config (d=64, residual_hidden=128, n_st_layers=2,
v45+v46+v50+v51+v52+v80 modules, train_samples=2048/epoch, batch 16,
val_stride=10, view-dropout 0.5 for v46, cosine LR). Only lr / weight decay /
early-stopping patience vary.

| Recipe | lr | weight decay | patience | Best val MPJPE (mm) | Best epoch | Trajectory (mm) | Outcome |
|---|---:|---:|:---:|---:|:---:|---|---|
| v1 (long, no reg) | 1e-3 | 0 | — | 65.28 | 2 | 80.93 → **65.28** → 103.38 → … → 323.55 (epoch 4, killed epoch 10 region) | sharp overfit |
| v2 | 5e-4 | 1e-4 | — | **39.70** | 2 | 75.51 → **39.70** → 168.32 (diverged) | killed at epoch 3 |
| v3 | 2e-4 | 5e-5 | 2 | 42.60 | 2 | 79.89 → **42.60** → 104.21 → 323.55 | early stop epoch 4 |
| v4 | 1e-4 | 2e-4 | 3 | 45.31 | 2 | 84.27 → **45.31** → 109.26 → 187.74 → 322.33 | early stop epoch 5 |

No NaN/inf in any run. Checkpoints/log fetched locally to
`outputs/a800_h36m_reg/` (v3 = `omniview_fusion_v80_h36m_true_gt_reg.*`,
v4 = `v4.*`, v2 epoch-2 best preserved on A800 as
`..._reg_epoch2best.pth`).

## Findings

1. **Every recipe overfits after epoch 2.** Lowering lr and raising weight
   decay only slows the post-epoch-2 collapse; it does not move the best
   epoch or close the gap. The sharp epoch-2 trough (~40 mm) followed by
   monotone degradation is recipe-invariant on this config.
2. **v80 best (39.70 mm) still loses to DLT (25.67 mm combined) by 1.5×**
   and to the Iskakov-style learned-weight baseline (23.38 mm) by 1.7×.
   Full-view v80 training has not beaten any geometric baseline on the
   repaired H36M protocol.
3. Consistent with the Shelf/Campus long-run conclusion: on true-GT labels
   the current v80 config memorizes train views rather than learning
   view-reliability that generalizes. Promising levers: view-dropout during
   training (v46 dropout is on, but the v80 weighting head may need stronger
   dropout / lower capacity), longer warmup, or training on mixed protocols.

## Evidence

| Run | Remote log | Local copy |
|---|---|---|
| v1 | `a800-D:/mnt/nvme0n1p1/zhangzy/motionflow-mv-h36m-truegt/outputs/omniview_fusion_v80_h36m_true_gt_long.log` | `outputs/a800_h36m_reg/omniview_fusion_v80_h36m_true_gt_long.log` |
| v2 | (same dir, killed mid-run; epoch-2 ckpt `..._reg_epoch2best.pth`) | — |
| v3 | `.../omniview_fusion_v80_h36m_true_gt_reg.log` | `outputs/a800_h36m_reg/omniview_fusion_v80_h36m_true_gt_reg.{log,pth,config.json}` |
| v4 | `.../omniview_fusion_v80_h36m_true_gt_reg.log` (rerun) | `outputs/a800_h36m_reg/v4.{log,pth,config.json}` |

Scripts: `scripts/run_v80_h36m_true_gt_long_a800.sh` (v1) and
`scripts/run_v80_h36m_true_gt_reg_a800.sh` (v2-v4, history in the header).

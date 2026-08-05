# Principal-Point Correction Ablation Log

## Goal
Validate whether an explicit, supervised principal-point (PP) correction layer can make the ray-attention multi-view pose estimator robust to small calibration errors in `cx, cy`, without degrading clean accuracy.

## Model
`RayAttentionFusionModelTemporalResidualPrincipalPoint`
- Input: 14-view MPI-INF-3DHP clips (T=13, J=28).
- Correction: bounded `tanh` offset in `[-20, 20]` px, predicted per view per frame from raw 2D observations + intrinsics.
- The corrected `K` is used for ray embedding and triangulation.

## Setup
- Venv: `/tmp/mf_venv` (native WSL, CUDA torch).
- GPU: RTX 4090 (local WSL).
- Training: 4 MPI-INF-3DHP sequences, val on S2/Seq1.

## Runs

### v1 (pp_loss_weight=1.0, sign bug)
- Clean MPJPE: ~11.27 mm.
- `cxcy_3px`: catastrophic (>1900 mm).
- Cause: sign error in supervision target and over-strong weight.

### v2 (pp_loss_weight=0.1, sign fixed) — complete
| Epoch | train_loss | val_MPJPE | note |
|-------|-----------:|----------:|------|
| 1 | 42.5368 | 14.41 mm | saved |
| 2 | 42.6501 | 16.16 mm |      |
| 3 | 42.5801 | 12.24 mm | saved |
| 4 | 42.4781 | 11.36 mm | saved |
| 5 | 42.4594 | 14.36 mm |      |
| 6 | 42.5509 | 10.88 mm | saved |
| 7 | 42.3722 | 11.56 mm |      |
| 8 | 42.4764 | 11.77 mm |      |
| 9 | 42.4421 | 11.29 mm |      |
| 10 | 42.5601 | **10.54 mm** | saved |

**Robustness (val S2/Seq1, best checkpoint):**
| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| clean | 10.46 | 7.02 |
| rot_0.5_deg | 17.76 | 9.55 |
| rot_1.0_deg | 29.96 | 14.48 |
| trans_5mm | 10.82 | 7.08 |
| trans_10mm | 11.99 | 7.36 |
| focal_1pct | 18.41 | 10.39 |
| focal_2pct | 29.97 | 14.45 |
| cxcy_3px | **13.84** | 7.66 |
| cxcy_5px | **17.05** | 8.25 |

**Takeaway:** Sign fix + weight 0.1 resolves the catastrophic `cxcy_3px/5px` failure. Explicit PP supervision successfully learns a bounded correction.

## Phase A ablation plan (ready to run)
Loss architecture / weight sweep on the small model:
- A1: 3D MSE only (baseline)
- A2: reprojection consistency only
- A3: balanced MSE+reproj
- A4: explicit offset (re-run, weight=0.1)
- A5: explicit + reprojection
- A6: higher explicit weight (0.5)

Scripts:
- `scripts/phase_a_ablation_runner.sh`
- `scripts/phase_a_eval_runner.sh`
- `scripts/self_evolve_pp_ablation.sh` (waits for v2, evaluates, then runs Phase A)

## Full model run
- d=64, residual_hidden=128, principal_point_hidden=64.
- 1000 clips/sequence, 5 epochs, val_stride=50.
- Best val MPJPE: **10.97 mm** (epoch 1).

Robustness (val S2/Seq1):
| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| clean | 10.97 | 6.66 |
| rot_0.5_deg | 17.79 | 9.22 |
| rot_1.0_deg | 29.59 | 14.78 |
| trans_5mm | 11.38 | 6.78 |
| trans_10mm | 12.36 | 7.08 |
| focal_1pct | 13.25 | 8.37 |
| focal_2pct | 23.02 | 11.17 |
| cxcy_3px | **13.03** | 6.71 |
| cxcy_5px | **15.26** | 7.41 |

Takeaway: full model slightly worse on clean MPJPE than small (10.97 vs 10.54), but better on focal-length and principal-point robustness. Likely needs more epochs/data to realize its full clean accuracy.

## Mixed-dataset run
- MPI-INF-3DHP + H36M, d=32, residual_hidden=64, principal_point_hidden=64.
- 500 clips/sequence, 10 epochs, val on MPI S2/Seq1.
- Best val MPJPE: **11.23 mm** (epoch 6).
- Clean eval: MPJPE=11.16 mm, PA-MPJPE=7.02 mm.
- Epoch results:
  | Epoch | train_loss | val_MPJPE |
  |-------|-----------:|----------:|
  | 1 | 25.9681 | 17.14 mm |
  | 2 | 26.1599 | 12.43 mm |
  | 3 | 26.0426 | 11.45 mm |
  | 4 | 26.0833 | 12.62 mm |
  | 5 | 26.1306 | 12.69 mm |
  | 6 | 26.1704 | 11.23 mm (saved) |
  | 7 | 26.1323 | 12.98 mm |
  | 8 | 26.0565 | 12.67 mm |
  | 9 | 26.1183 | 11.82 mm |
  | 10 | 25.9943 | 11.65 mm |

Note: mixed training lags MPI-only (11.16 vs 10.46 mm) but demonstrates cross-dataset generalization. Future work: increase MPI proportion or model capacity.

## Mixed-dataset preparation
- Ported `PrincipalPointCorrection` to the mixed-dataset residual model (`RayAttentionFusionModelTemporalMixedResidualPrincipalPoint`).
- New training script: `experiments/train_mixed_dataset_principal_point.py`.
- New eval script: `experiments/eval_mixed_dataset_principal_point.py`.
- H36M WebBridge data converted from mm to meters (`data/webbridge/h36m_meters/`) for consistent units with MPI-INF-3DHP.
- Run script: `scripts/run_mixed_pp_wsl.sh`.

## Phase A early findings
- A1 (3D MSE only) still running when Phase A was stopped; slow due to sequential runner.
- A5 (explicit + reprojection, weight=0.5) failed: loss exploded to ~1e12, val_MPJPE >500 mm. Scale analysis shows reprojection MSE is ~3.5M× larger than 3D MSE for the same geometry; a useful weight would be ~1e-6, not 0.5.
- Decision: use v2 (explicit PP only, weight=0.1) as the best small configuration and scale it to full model and mixed data.

## Focal-aware intrinsic correction (in progress)
- Added `max_focal_scale` to `PrincipalPointCorrection`; MLP output dim grows from 2 to 3 and predicts `(dx, dy, s)`.
- Supervision: predicted focal scale targets `1 / true_focal_scale`.
- Functional construction of corrected `K` avoids autograd in-place errors.
- Training command (WSL RTX 4090):
  ```bash
  python experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz ... s_03_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
    --epochs 10 --train_samples 500 --val_stride 50 --batch_size 8 \
    --pp_loss_weight 0.1 --focal_max_scale 0.1 \
    --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 5.0
  ```
- Best val MPJPE: **12.90 mm** (epoch 10), clean eval: MPJPE 12.82 mm / PA-MPJPE 9.36 mm.
- Robustness (vs PP-only v2 in parentheses):
  | Condition | MPJPE (mm) | PA-MPJPE (mm) |
  |---|---:|---:|
  | clean | 12.82 (10.54) | 9.36 (7.02) |
  | rot_0.5_deg | 18.07 (17.76) | 10.81 (9.55) |
  | rot_1.0_deg | 28.94 (29.96) | 14.23 (14.48) |
  | focal_1pct | **18.29** (18.41) | 10.35 (10.39) |
  | focal_2pct | **28.42** (29.97) | 12.65 (14.45) |
  | cxcy_3px | 14.31 (13.84) | 8.88 (7.66) |
  | cxcy_5px | **16.51** (17.05) | 8.86 (8.25) |
- Observations:
  - Focal-length robustness improves modestly at 1%/2% focal error (18.41→18.29, 29.97→28.42), validating the idea.
  - Clean accuracy drops by ~2.3 mm vs PP-only, and principal-point robustness is slightly worse.
  - The focal correction task is harder with the small model; extra capacity or a tighter focal_max_scale may restore clean accuracy.

### Full focal-aware model (d=64, h=128, focal_max_scale=0.1)
- Best val MPJPE: **14.71 mm** (epoch 5). Clean eval: 14.88 mm / PA-MPJPE 9.37 mm.
- Robustness:
  | Condition | MPJPE (mm) | PA-MPJPE (mm) |
  |---|---:|---:|
  | clean | 14.88 | 9.37 |
  | focal_1pct | 21.97 | 10.89 |
  | focal_2pct | 32.48 | 13.71 |
  | cxcy_3px | 16.00 | 9.59 |
  | cxcy_5px | 17.79 | 10.06 |
- Full model after 15 total epochs:
  - Best val MPJPE: **12.11 mm**, clean eval: 12.21 mm / PA-MPJPE 6.94 mm.
  - Robustness:
    | Condition | MPJPE (mm) | PA-MPJPE (mm) |
    |---|---:|---:|
    | clean | 12.21 | 6.94 |
    | focal_1pct | 20.24 | 9.42 |
    | focal_2pct | 31.04 | 12.81 |
    | cxcy_3px | 12.91 | 7.21 |
    | cxcy_5px | 14.40 | 7.69 |
- Observations:
  - Full focal-aware model still underperforms PP-only full model on clean (12.21 vs 10.97 mm) and focal robustness (focal_1pct 20.24 vs 13.25 mm; focal_2pct 31.04 vs 23.02 mm).
  - Principal-point robustness is slightly better (cxcy_3px 12.91 vs 13.03; cxcy_5px 14.40 vs 15.26).
  - Shared PP/focal MLP and identical loss weight may under-resource the focal task; a dedicated focal head or separate loss weight may help.
  - Next: try separate focal loss weight and/or mixed-dataset training to improve generalization.

## Key hypotheses
1. Explicit PP supervision alone can teach the correction head, but too high a weight harms the main task.
2. Adding reprojection consistency requires careful normalization; current implementation explodes with weight 0.5.
3. With v2 working, scale to d=64/h=128 and mixed WebBridge datasets.
4. Extending the correction layer to focal length should improve focal_1pct/focal_2pct without degrading clean accuracy.

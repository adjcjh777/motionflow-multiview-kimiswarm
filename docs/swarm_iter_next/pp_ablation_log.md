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

### Mixed-dataset focal-aware (MPI + H36M)
- Best val MPJPE: **13.59 mm**, clean eval: 13.77 mm / PA-MPJPE 8.50 mm.
- Worse than MPI-only focal small (12.90 mm val) and PP-only small (10.54 mm clean).
- Mixed training is harder and likely needs a dedicated focal head or smaller focal loss weight.

### Baseline comparison (no intrinsic correction)
- Perturbation-trained residual small (d=32, h=64): clean 14.97 mm, focal_1pct 14.95 mm, focal_2pct 15.35 mm, cxcy_3px 1592.69 mm, cxcy_5px 1894.61 mm.
- Confirms the learned correction layer removes the catastrophic principal-point failure.

### Dedicated focal head (small MPI-only)
- Best val MPJPE: **12.73 mm**, clean eval: 12.81 mm / PA-MPJPE 9.80 mm.
- Robustness vs shared head:
  | Condition | Dedicated head | Shared head |
  |---|---:|---:|
  | clean | 12.81 | 12.82 |
  | focal_1pct | 20.25 | 18.29 |
  | focal_2pct | 31.40 | 28.42 |
  | cxcy_3px | 13.89 | 14.31 |
  | cxcy_5px | 16.21 | 16.51 |
- Observations:
  - Dedicated focal head did **not** improve focal-length robustness; it made it slightly worse.
  - Principal-point robustness marginally improved.
  - The problem is likely not head capacity. Possible causes: focal perturbation std (1%) is too small to learn a strong signal, or the correction should operate on normalized intrinsics / camera-independent features.

### High focal perturbation (cam_aug_focal=0.05, focal_max_scale=0.2)
- Epoch 1: val_MPJPE = 25.89 mm, Epoch 2: 43.67 mm.
- Validation diverges immediately; 5% focal perturbation is too strong for the current correction mechanism.
- Conclusion: simply increasing focal perturbation does not solve the problem.

## Key hypotheses
1. Explicit PP supervision alone can teach the correction head, but too high a weight harms the main task.
2. Adding reprojection consistency requires careful normalization; current implementation explodes with weight 0.5.
3. With v2 working, scale to d=64/h=128 and mixed WebBridge datasets.
4. Extending the correction layer to focal length should improve focal_1pct/focal_2pct without degrading clean accuracy.

## Cross-view temporal residual + principal-point correction
- After focal-aware experiments underperformed PP-only, pivot to combining the strongest clean model (cross-view temporal residual, ~10.2 mm) with PP correction.
- New training script: `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`.
- New eval script: `experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`.
- WSL runner: `scripts/run_crossview_pp_small_wsl.sh`.
- Small model: d=32, residual_hidden=64, n_st_layers=2, pp_loss_weight=0.1, cam_aug_pp=5.0, cam_aug_focal=0.01, 10 epochs, train on S1/S3, val on S2/Seq1.

### Results (small model)
- Best val MPJPE: **10.94 mm** (epoch 5).
- Clean eval: MPJPE=10.97 mm, PA-MPJPE=7.97 mm.
- Calibration robustness:
  | Condition | MPJPE (mm) | PA-MPJPE (mm) |
  |---|---:|---:|
  | clean | 10.97 | 7.97 |
  | rot_0.5_deg | 17.20 | 10.52 |
  | rot_1.0_deg | 29.65 | 15.79 |
  | trans_5mm | 11.49 | 8.07 |
  | trans_10mm | 12.23 | 8.12 |
  | focal_1pct | 19.88 | 11.04 |
  | focal_2pct | 31.73 | 14.98 |
  | cxcy_3px | **13.77** | 8.90 |
  | cxcy_5px | **16.67** | 9.60 |

### Observations
- Clean accuracy (10.97 mm) is slightly worse than the no-PP cross-view residual baseline (~10.20 mm) and the PP-only temporal baseline (10.54 mm).
- Principal-point robustness is strong: cxcy_3px only rises to 13.77 mm, cxcy_5px to 16.67 mm, confirming the correction layer works.
- The cross-view model appears to need more capacity or a different PP feature representation to match its no-PP clean accuracy while retaining PP robustness.

### Full model results (d=64, residual_hidden=128)
- Best val MPJPE: **10.09 mm** (epoch 8).
- Clean eval: MPJPE=10.09 mm, PA-MPJPE=5.00 mm.
- Calibration robustness:
  | Condition | MPJPE (mm) | PA-MPJPE (mm) |
  |---|---:|---:|
  | clean | 10.09 | 5.00 |
  | rot_0.5_deg | 16.89 | 8.11 |
  | rot_1.0_deg | 27.45 | 13.50 |
  | trans_5mm | 10.61 | 5.20 |
  | trans_10mm | 11.23 | 5.44 |
  | focal_1pct | 19.13 | 8.07 |
  | focal_2pct | 30.41 | 12.18 |
  | cxcy_3px | **11.41** | 5.75 |
  | cxcy_5px | **13.87** | 6.61 |

### Observations
- Full model improves over small model on clean (10.09 vs 10.97 mm) and all robustness conditions, especially principal-point drift (cxcy_3px 11.41 vs 13.77 mm).
- Clean accuracy now matches or exceeds the no-PP cross-view residual small baseline (~10.20 mm), while adding PP robustness.
- PA-MPJPE drops dramatically from 7.97 mm (small) to 5.00 mm (full), indicating better pose alignment.

### Human3.6M results (d=32, residual_hidden=64)
- Best val MPJPE: **6.16 mm** (epoch 6). Clean eval: MPJPE=6.20 mm, PA-MPJPE=4.26 mm.
- Calibration robustness:
  | Condition | MPJPE (mm) | PA-MPJPE (mm) |
  |---|---:|---:|
  | clean | 6.20 | 4.26 |
  | rot_0.5_deg | 10.35 | 9.29 |
  | rot_1.0_deg | 19.65 | 17.47 |
  | trans_5mm | 8.68 | 4.60 |
  | trans_10mm | 13.32 | 5.84 |
  | focal_1pct | 9.75 | 6.63 |
  | focal_2pct | 15.50 | 10.20 |
  | cxcy_3px | 16.20 | 4.87 |
  | cxcy_5px | 25.04 | 6.73 |
- Observations: clean accuracy is excellent, but principal-point robustness is worse than on MPI (4 views vs 14 views), likely because fewer views provide less redundancy for correcting PP drift.

### Full H36M results (d=64, residual_hidden=128)
- Best val MPJPE: **5.24 mm** (epoch 7). Clean eval: MPJPE=5.24 mm, PA-MPJPE=4.84 mm.
- Calibration robustness:
  | Condition | MPJPE (mm) | PA-MPJPE (mm) |
  |---|---:|---:|
  | clean | 5.24 | 4.84 |
  | rot_0.5_deg | 10.66 | 8.94 |
  | rot_1.0_deg | 21.98 | 19.66 |
  | trans_5mm | 7.44 | 5.25 |
  | trans_10mm | 12.65 | 5.90 |
  | focal_1pct | 8.87 | 7.36 |
  | focal_2pct | 14.62 | 10.84 |
  | cxcy_3px | 15.17 | 8.02 |
  | cxcy_5px | 23.86 | 10.31 |
- Observations: clean accuracy improves further, but principal-point robustness remains weaker than MPI. The 4-view H36M setup has too little redundancy for strong PP correction.

### pp_loss_weight ablation (MPI)
| Model | pp_loss_weight | Best val MPJPE | Clean MPJPE | Clean PA-MPJPE | cxcy_3px | cxcy_5px |
|---|---:|---:|---:|---:|---:|---:|
| small | 0.10 | 10.94 | 10.97 | 7.97 | 13.77 | 16.67 |
| small | **0.05** | **10.30** | **10.34** | **6.28** | **11.29** | **13.13** |
| full | 0.10 | 10.09 | 10.09 | 5.00 | 11.41 | 13.87 |
| **full** | **0.05** | **9.41** | **9.41** | **5.66** | **10.93** | **13.47** |

Reducing the PP supervision weight from 0.1 to 0.05 improves both clean accuracy and principal-point robustness, suggesting 0.1 was overly regularizing the correction head. The full model with pp_loss_weight=0.05 reaches the best clean MPJPE (9.41 mm) and strong PP robustness.

### Longer training (full MPI, pp_loss_weight=0.05)
| Epochs | Best val MPJPE | Clean MPJPE | Clean PA-MPJPE | cxcy_3px | cxcy_5px |
|---:|---:|---:|---:|---:|---:|
| 10 | 9.41 | 9.41 | 5.66 | 10.93 | 13.47 |
| **20** | **9.32** | **9.32** | **5.37** | **11.18** | **13.78** |

Training for 20 epochs gives a marginal clean improvement (9.41 → 9.32 mm) but slightly worse principal-point robustness, suggesting the model begins to overfit or the optimal stopping point is around 10–16 epochs.

### Next steps
1. Scale to mixed-dataset training (MPI-INF-3DHP + H36M) for cross-dataset generalization.
2. Consider predicting PP correction from pooled spatio-temporal features for a tighter integration.
3. Investigate H36M-specific data preprocessing or view selection to improve PP robustness with fewer views.

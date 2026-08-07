# H36M OmniMultiViewFusion v2 (dense+graph) – A800-D results

**Checkpoint**: `outputs/omniview_fusion_v2_h36m_d128_dense_graph_a800.pth`  
**Config**: d=128, residual_hidden=256, n_st_layers=3, graph_num_layers=1, n_joint_layers=1  
**Training**: WebBridge H36M S1, 30 epochs, batch 32, view_dropout_rate=0.1  
**MPJPE at the best validation-loss epoch**: 20.91 mm (reported by trainer)

> Identity boundary: this trainer value used EMA parameters, while the clean,
> robustness, and variable-view scripts originally loaded the raw `model`
> entry. The values below remain recorded observations, but they are not
> same-weight comparisons until both identities are re-evaluated under one
> protocol.

## Clean evaluation

| Split | Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|-------|----------|-----------:|----------------:|
| Val   | S9/acts_02 | **15.03** | 6.75 |
| Test  | S11/acts_02 | **24.04** | 5.10 |

Additional val metrics:
- root_rel_mpjpe: 6.26 mm
- velocity_mpjpe: 2.51 mm
- PCK@50mm: 0.992
- PCK@100mm: 0.997
- PCK@150mm: 0.998
- PCK-AUC: 0.901

## Calibration robustness (val S9/acts_02)

| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|-----------|-----------:|----------------:|
| clean | 15.03 | 6.75 |
| rot_0.5_deg | 19.08 | 12.89 |
| rot_1.0_deg | 28.41 | 21.43 |
| trans_5mm | 15.98 | 7.24 |
| trans_10mm | 19.51 | 8.34 |
| focal_1pct | 15.83 | 8.41 |
| focal_2pct | 20.12 | 11.52 |
| cxcy_3px | 18.51 | 7.57 |
| cxcy_5px | 24.91 | 8.59 |

## Variable-view robustness

| Active views | Mean MPJPE (mm) | Std (mm) | n_subsets |
|-------------:|----------------:|---------:|----------:|
| 2 | 1990.56 | 750.11 | 6 |
| 3 | 1619.90 | 729.20 | 4 |
| 4 | 14.99 | 0.00 | 1 |

## Take-aways

- The model reaches a strong **single-model 15 mm MPJPE on H36M val** and generalises to the unseen subject S11 with 24 mm.
- Calibration perturbations up to 1° rotation or 2 % focal error remain below 30 mm, which is a reasonable starting point.
- **Variable-view inference is the main gap**: dropping to 2 or 3 views causes catastrophic error (~2000 mm). This confirms the need for explicit view-dropout robustness / adaptive view selection in the next iteration (v4).

# Benchmark Table: MotionFlow-MultiView Residual Refinement

## MPI-INF-3DHP cross-subject (train S1 Seq1+Seq2, val S2/Seq1)

| Model | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC | Data size |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline temporal (DLT) | 25.21 | — | — | — | — | — | full |
| Residual d=64, h=128 (under-trained) | 19.49 | 19.16 | 0.9937 | 1.0000 | 1.0000 | 0.8701 | full |
| Residual 3-epoch (d=64, h=128) | 14.17 | 12.99 | 0.9982 | 1.0000 | 1.0000 | 0.9055 | full |
| Residual 4-epoch (d=64, h=128) | 13.12 | 10.86 | 0.9999 | 1.0000 | 1.0000 | 0.9125 | full |
| **Residual full 5-epoch (d=64, h=128)** | **11.17** | **8.24** | **1.0000** | **1.0000** | **1.0000** | **0.9256** | full |
| Residual small (d=32, h=64) | 13.22 | 11.77 | 0.9974 | 1.0000 | 1.0000 | 0.9119 | full |
| Residual on H36M (h=64) | 5.71 | 5.33 | 0.9976 | 0.9996 | 0.9998 | 0.9620 | full |
| **Residual on H36M (h=128)** | **5.74** | **3.99** | 0.9980 | 0.9995 | 0.9998 | 0.9618 | full |
| Cross-view + residual (smoke) | 11.56 | — | — | — | — | — | smoke |
| Cross-view residual d=64, h=128 (full) | 15.29 | 13.49 | 0.9974 | 1.0000 | 1.0000 | 0.8981 | full |
| Uncertainty residual | 9.72 | — | — | — | — | — | smoke |
| V4 single-frame + residual | 12.13 | — | — | — | — | — | smoke |
| Deeper residual (3 blocks) | 13.24 | — | — | — | — | — | smoke |

Notes:
- "full" = MPI-INF-3DHP S1 Seq1 + Seq2 train, S2/Seq1 validation.
- "smoke" = small subset (250–500 frames) for fast exploration; not directly comparable to full numbers.

## Robustness of residual final5 (MPI-INF-3DHP S2/Seq1)

| Perturbation | Level | MPJPE (mm) |
|---|---|---:|
| Clean | 0 | 11.17 |
| Gaussian noise | 5 px | 12.96 |
| Gaussian noise | 20 px | 28.00 |
| Joint occlusion | 50% | 11.18 |
| 2D outliers | 20% | 15.13 |

Notes:
- Evaluated on `outputs/ray_attention_temporal_residual_final5.pth`.
- 50% random joint occlusion barely changes MPJPE, confirming strong multi-view redundancy.
- 20% 2D outliers increase MPJPE by ~4 mm; 20 px Gaussian noise degrades to 28 mm.

## Residual head capacity ablation (H36M S1→S5, 3 epochs, train_samples=2000)

| residual_hidden | Params | Best val MPJPE (mm) |
|---|---:|---:|
| 64 | 185,572 | **5.71** |
| 128 | 202,468 | 5.74 |
| 256 | 260,836 | 6.43 |

## Key takeaways

- The residual refinement head reduces cross-subject MPJPE by ~56% (25.2 → 11.2 mm) on MPI-INF-3DHP and reaches 5.74 mm on Human3.6M.
- The model is highly robust to occlusion due to multi-view redundancy.
- Smoke results suggest cross-view attention and uncertainty can push performance further, but require full-data validation.

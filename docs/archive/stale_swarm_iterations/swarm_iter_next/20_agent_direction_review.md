# 20-Agent Direction Review — Next Iteration Roadmap

**Date:** 2026-08-06
**Baseline:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
**Current best:** MPI-INF-3DHP clean **9.32 mm** / PA-MPJPE **5.37 mm**
**GPU queue:** variable-view eval → visibility-gated v2 → SSL pre-training → spatiotemporal PP

This document summarizes a parallel 20-agent exploration of the next-iteration
plan from `docs/next_iteration_plan_swarm.md`. Each direction was reviewed for
implementation status, concrete next steps, expected benefit/risk, and GPU-queue
compatibility.

## Quick status matrix

| # | Direction | Priority | Status | Next deliverable |
|---|-----------|----------|--------|------------------|
| 1 | Multi-view SSL / masked pre-training | P1 | Skeleton + smoke ready, no full run | `outputs/ray_attention_ssl_h36m.pth` |
| 2 | Visibility-aware adaptive fusion | **P0** | Full skeleton, training queued | `outputs/ray_attention_temporal_crossview_residual_visibility_v2_mpiinf3dhp.pth` |
| 3 | Cross-view spatio-temporal Transformer (T×V×J) | P1 | Model + trainer + smoke ready, untrained | `outputs/spatiotemporal_principal_point_mpiinf3dhp.pth` |
| 4 | Camera calibration robustness (PP/focal/distortion/extrinsic) | **P0** | PP head works for small perturbations, PP 10 px catastrophic | Robustness matrix with clean ≤ 9.6 mm, pp_10px < 15 mm |
| 5 | Temporal consistency / long-term dependencies | P1 | Long-clip + velocity script queued, not run | `outputs/*velocity_longclip*.pth` |
| 6 | Multi-scale / multi-resolution spatial features | P1 | `SpatialFeaturePyramid` exists, unwired from best model | GPU smoke with pyramid PP |
| 7 | WebBridge integration & cleaning | P1 | Loader/benchmark exist, H36M S9/S11 partial | Unified cross-dataset table |
| 8 | Variable-view inference & view dropout | **P0** | **Running now** on curriculum checkpoint | `outputs/variable_views_curriculum_final.json` |
| 9 | Uncertainty quantification & confidence fusion | P1 | Uncertainty model exists, no full benchmark | Full MPI eval + robustness report |
| 10 | Graph neural networks for skeleton fusion | P1 | Graph module exists, no runnable best-model training | PP-graph GPU smoke |
| 11 | Physics / kinematic consistency (GN/bone-length) | P1 | GN triangulation + bone-loss exist, unwired from best model | GN-PP or bone-loss-PP checkpoint |
| 12 | Cross-dataset domain adaptation (GRL+FiLM) | P1 | Wrapper + smoke exist, missing canonical skeleton map | `skeleton_maps.py` + cross-dataset trainer |
| 13 | Real-time inference optimization | P2 | Profiling scripts exist, SDPA/FlashAttention not implemented | GPU latency baseline + SDPA smoke |
| 14 | Occlusion / partial visibility | P1 | Visibility v2 + synthetic occlusion tools exist, no real-data eval | Occlusion robustness table |
| 15 | Self-supervised / masked pre-training | P1 | (Same as #1) | H36M SSL checkpoint + data-efficiency curve |
| 16 | Multi-person association | P2 | Synthetic smoke passes, geometric solver missing | Hungarian geometric association + metric |
| 17 | Action semantics / category prior | P2 | Action-aware PP model skeleton exists, no full training | H36M per-action ablation |
| 18 | 3D Gaussian splatting / novel-view regularizer | P2 | CPU smoke passes, no loss module | `GaussianSplattingLoss` + synthetic ablation |
| 19 | Interpretability & failure analysis | **P0** | PP failure script exists, not scaled to full test set | Full MPI failure profile + paper figures |
| 20 | Evaluation protocol, metrics & reproducibility | **P0** | `BenchmarkProtocol` + metrics ready, no multi-seed manifest | Canonical splits + `manifest.json` |

## Immediate P0 actions (next 1–2 GPU/CPU cycles)

1. **Finish the running variable-view evaluation** (`outputs/variable_views_curriculum_final.json`).
2. **Run visibility-gated v2 training** once the GPU is free.
3. **Diagnose/fix PP 10 px catastrophic failure** in `principal_point_correction.py`.
4. **Scale failure analysis** to the full MPI-INF-3DHP test set and produce paper figures.
5. **Set up canonical splits + multi-seed manifest** in `BenchmarkProtocol`.

## GPU queue recommendation (single RTX 4090)

1. Variable-view eval (running)
2. Visibility-gated v2 (already queued)
3. SSL pre-training on H36M (already queued)
4. Spatiotemporal PP (already queued)
5. Calibration robustness re-train (focal + stronger PP curriculum)
6. Uncertainty / graph / pyramid ablations (P1)

## CPU work that can run in parallel now

- Canonical skeleton mapping (`motionflow_mv/data/skeleton_maps.py`)
- Failure-analysis scaling and visualization scripts
- `tests/test_ssl_dataset.py`, `tests/test_failure_analysis_pp.py`
- Benchmark protocol wiring for `eval_full_metrics.py`
- SDPA/FlashAttention smoke implementation

## Notes for ICRA/CVPR 2027

- The strongest publishable narrative remains: **camera-agnostic multi-view fusion + calibration/visibility robustness + variable-view inference**.
- Cross-dataset generalization and SSL data-efficiency are the next most compelling supporting stories.
- Real-time optimization, multi-person association, and Gaussian regularizer are valuable but secondary until the P0/P1 accuracy story is locked.

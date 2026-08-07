# 20-Agent Swarm Synthesis — 2026-08-07

Status: MPI-INF-3DHP S2/Seq1 ensemble **8.61 mm** MPJPE (below 8.75 mm threshold).

## Current strongest result

- Ensemble of `bayesian_tri_v2_stabilized` + `bayesian_tri_v2_aug` on d=128 Bayesian Tri v2.
- Single best model: `bayesian_tri_v2_stabilized` at **9.03 mm**.
- Extended robustness shows largest gaps under **view dropout** (18.15 mm at 30%) and **joint occlusion** (16.99 mm at 30%).

## Top-ranked next experiments (ROI order)

| Rank | Direction | Concrete action | Success metric |
|---|---|---|---|
| 1 | **Visibility-gated fusion v2 on Bayesian Tri v2** | Subclass `RayAttentionFusionModelBayesianTriV2` with a learned per-view/per-joint visibility head and BCE loss; warm-start from 9.03 mm checkpoint. | Single-model clean ≤9.0 mm **and** `view_dropout_30` ≤16.3 mm |
| 2 | **Graph-joint residual refiner on Bayesian Tri v2** | Replace the residual MLP with `SkeletonGraphResidualRefiner`; keep triangulation/GN unchanged. | Single-model clean <9.03 mm, ideally ≤8.90 mm |
| 3 | **Kinematic-chain refiner on Bayesian Tri v2** | Apply `KinematicChainGraphRefinerTemporal` to final 3D output with small auxiliary loss. | Limb joints improve ≥5% without trunk regression |
| 4 | **Extended camera-perturbation curriculum** | Ramp rot/focal/PP augmentation; warm-start from 9.03 mm checkpoint. | `rot_0.5°` ≤14 mm or `focal_1%` ≤15 mm, clean ≤9.2 mm |
| 5 | **Focal-scale correction head** | Enable `focal_max_scale=0.05` and `focal_loss_weight=0.2` in existing trainer. | `focal_1%` improves ≥1 mm, clean preserved |
| 6 | **Attention-entropy regularisation** | Add entropy penalty on normalised triangulation weights. | Clean ≤9.03 mm **and** `view_dropout_30` improves ≥1 mm |
| 7 | **Epipolar-bias-v2-lite full-data** | Already queued (`scripts/run_epipolar_bias_v2_lite_pp_full_data_wsl.sh`). | Single-model MPJPE <9.10 mm |
| 8 | **Mixed-dataset MPI+H36M+AIST** | Re-run mixed trainer with unit-corrected H36M meters and larger capacity. | MPI ≤9.5 mm, H36M ≤15 mm, AIST ≤15 mm |
| 9 | **Rotation correction head** | Predict bounded `so(3)` residual per view before triangulation. | `rot_0.5°` ≤13 mm, `rot_1.0°` ≤20 mm |
| 10 | **Variable-view MPJPE@k curve** | Evaluate best d=128 model for k=2..14. | Paper figure ready |

## Cross-cutting infrastructure needs

- MPI-INF-3DHP official **test subjects TS1-TS6** for test-set evaluation.
- Repeated-seed runs (3–5 seeds) for final paper statistics.
- Runtime/latency benchmark on RTX 4090 for the final model.

## Immediate next action

Implement **#1 Visibility-gated fusion v2 on Bayesian Tri v2** so it is ready to run when GPU capacity frees up.

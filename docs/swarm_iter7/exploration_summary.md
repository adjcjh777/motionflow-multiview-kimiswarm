# Exploration Summary: Swarm Iteration 7 — 20-Agent Next-Step Sprint

**Date:** 2026-08-06
**Trigger:** Continue iterative exploration; design a more complex yet minimal-viable multi-view fusion model and supporting experiments.

## Constraints in force

- WSL RTX 4090 is running cross-view PP curriculum + view-dropout training; no new GPU jobs started.
- A800-D is read-only; no writes or training there.
- Each direction must produce the smallest CPU-safe artifact or a queued GPU skeleton.

## 20 directions and deliverables

| # | Direction | Deliverable | GPU/CPU | Status |
|---|-----------|-------------|---------|--------|
| 1 | Multi-view SSL masked-view reprojection | `docs/swarm_iter7/multi_view_ssl_masked_reprojection.md` + smoke pretraining script | GPU skeleton + CPU smoke | Done |
| 2 | Visibility-aware adaptive fusion | `docs/swarm_iter7/visibility-aware-adaptive-fusion.md` + synthetic occlusion smoke | CPU smoke passed | Done |
| 3 | Cross-view spatio-temporal Transformer (T×V×J) | `docs/swarm_iter7/cross_view_spatio_temporal_transformer_txvxj.md` + CPU shape/grad check | GPU skeleton | Done |
| 4 | Camera calibration robustness | `docs/swarm_iter7/camera_calibration_robustness_focal_distortion_extrinsic_curriculum.md` + perturb diagnostic | CPU diagnostic run | Done |
| 5 | Temporal consistency / long-term dependencies | `docs/swarm_iter7/temporal_consistency_long_term_dependencies.md` + velocity loss module + CPU verification | GPU launcher | Done |
| 6 | Multi-scale spatial features | `docs/swarm_iter7/ablate_multiscale_spatial_features_in_pp_model.md` + pyramid smoke | CPU smoke passed | Done |
| 7 | WebBridge integration & cleaning | `docs/swarm_iter7/webbridge_integration_data_cleaning.md` + H36M S9/S11 meter converter | CPU converter run | Done |
| 8 | Variable-view inference & view dropout | `docs/swarm_iter7/variable_view_inference_and_view_dropout.md` + smoke results JSON | CPU smoke passed | Done |
| 9 | Uncertainty & confidence fusion | `docs/swarm_iter7/uncertainty_quantification_and_confidence_fusion.md` + CPU sanity check | GPU skeleton | Done |
| 10 | Graph neural networks for skeleton fusion | `docs/swarm_iter7/graph_neural_networks_skeleton_fusion.md` + GNN model skeleton + test | CPU test passed | Done |
| 11 | Physics / focal self-calibration | `docs/swarm_iter7/physics_kinematic_focal_self_calibration.md` + focal smoke + launcher | CPU smoke passed | Done |
| 12 | Cross-dataset domain adaptation | `docs/swarm_iter7/cross_dataset_domain_adaptation.md` + CPU smoke trainer | CPU smoke run | Done |
| 13 | Real-time inference optimization | `docs/swarm_iter7/real_time_inference_optimization.md` + CPU profile + torch.compile probe | CPU profile | Done |
| 14 | Occlusion / partial visibility | `docs/swarm_iter7/occlusion_partial_visibility_handling.md` + v2 occlusion smoke | CPU smoke passed | Done |
| 15 | Self-supervised / masked pre-training | `docs/swarm_iter7/self_supervised_masked_pretraining_protocol.md` + data-efficiency split generator | CPU generator run | Done |
| 16 | Multi-person association | `docs/swarm_iter7/multi_person_association.md` + smoke results | CPU smoke passed | Done |
| 17 | Action semantics / category prior | `docs/swarm_iter7/action_semantics_category_prior.md` + action-aware model + CPU smoke | CPU smoke passed | Done |
| 18 | 3D Gaussian splatting regularizer | `docs/swarm_iter7/3d_gaussian_splatting_novel_view_regularizer.md` + Gaussian smoke | CPU smoke passed | Done |
| 19 | Interpretability & failure analysis | `docs/swarm_iter7/interpretability_failure_analysis.md` + failure correlation script | CPU analysis run | Done |
| 20 | Evaluation protocol & reproducibility | `docs/swarm_iter7/evaluation_protocol_metrics_reproducibility.md` + multi-seed launcher + tests | CPU tests passed | Done |

## Immediate winners (P0 to queue on GPU once curriculum finishes)

1. **Direction 4 — Camera calibration robustness** (focal self-calibration + extrinsic curriculum): directly addresses the worst robustness failures.
2. **Direction 2 — Visibility-aware adaptive fusion**: next in the existing GPU queue; code already ready.
3. **Direction 3 — Spatio-temporal (T×V×J) Transformer**: the requested "more complex multi-view model"; factorized design keeps compute manageable.
4. **Direction 5 — Temporal consistency / velocity loss**: low-risk add-on to the PP trainer; can be combined with direction 4.
5. **Direction 7 — WebBridge H36M test benchmark**: needed for cross-dataset table in the paper.

## Integration plan

1. After the running curriculum finishes, evaluate it (clean + robustness).
2. Run visibility v2 training (already queued).
3. Queue the **focal self-calibration + velocity loss** variant as a single follow-up to the PP curriculum.
4. Queue the **spatiotemporal (T×V×J) factorized model** as the first genuinely more complex architecture.
5. Continue with SSL pre-training and the H36M test benchmark.

## Blockers

- GPU is single-threaded; all GPU experiments must wait for the current queue.
- `torch.compile` on WSL fails due to missing C++ compiler; real-time optimization direction remains a CPU-only profile.
- Some parallel agents created overlapping staged files; history is slightly interleaved but consistent.

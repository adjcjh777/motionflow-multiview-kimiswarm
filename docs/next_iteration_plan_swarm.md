# Next Iteration Plan — 20-Agent Swarm Synthesis

**Date:** 2026-08-05
**Baseline model:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
**Current best result:** MPI-INF-3DHP clean **9.32 mm** / PA-MPJPE **5.37 mm**

## Constraints

- WSL RTX 4090 正在运行 cross-view PP curriculum + view-dropout 训练，新 GPU 实验需排队。
- A800-D 为只读存储，不能写入数据或启动训练。
- 坚持最小可行验证，避免堆叠多个未验证模块。

## 20 directions at a glance

| # | Direction | Priority | Summary | Key opportunity | Next step |
|---|-----------|----------|---------|-----------------|-----------|
| 1 | Multi-view pre-training representations | P1 | Masked-view reprojection pre-training on H36M/AIST++/synthetic videos, then fine-tune on MPI. | Data-efficiency story; less reliance on small MPI labels. | Implement `ssl_dataset.py` + `pretrain_ray_attention_ssl.py`. |
| 2 | Visibility-aware adaptive fusion | **P0** | Add explicit visibility head to best PP model; soft-gate DLT weights. | Occlusion robustness; complements variable-view. | Run `scripts/run_crossview_pp_visibility_wsl.sh` when GPU is free. |
| 3 | Cross-view spatio-temporal Transformer | P1 | Feed PP correction into a full `(T×V×J)` Transformer or factorized variant. | Potential clean MPJPE below 9 mm. | Smoke-test `spatiotemporal_principal_point_model.py`. |
| 4 | Camera calibration robustness | **P0** | Extend PP correction to focal length / distortion; stronger extrinsic curriculum. | Addresses biggest current weakness (rot_0.5° → 16.89 mm, focal_1% → 19.13 mm). | Evaluate current curriculum; then add focal loss / stronger perturbation. |
| 5 | Temporal consistency / long-term dependencies | P1 | Longer clips + velocity smoothness or multi-scale temporal conv. | Reduce jitter; possibly clean < 9.0 mm. | Train with `clip_len=25` + velocity loss. |
| 6 | Multi-scale / multi-resolution spatial features | P1 | Spatial feature pyramid over joints, coarse-to-fine. | Better distal-joint robustness. | Add `SpatialFeaturePyramid` module + ablation. |
| 7 | WebBridge integration & cleaning | P1 | Unified benchmark, audit `.npz` quality, fix H36M S9/S11. | Prerequisite for cross-dataset tables; diagnose 101 mm H36M failure. | Run `run_webbridge_benchmark.py` + `audit_webbridge_npz.py`. |
| 8 | Variable-view inference & view dropout | **P0** | Run fixed-slot model with 2–14 active views; train with view dropout. | Practical deployment with arbitrary camera counts. | Run `eval_variable_views.py` on best PP checkpoint. |
| 9 | Uncertainty quantification & confidence fusion | P1 | Per-view log-variance head weighted into DLT. | Interpretable view confidence; clean + robust gains. | Implement uncertainty PP model + 10-epoch smoke. |
| 10 | Graph neural networks for skeleton fusion | P1 | Replace dense joint attention with `GraphJointRelation`. | Skeleton-aware reasoning + anatomy. | Warm-start small ablation. |
| 11 | Physics / kinematic consistency | P1 | Focal self-calibration or Gauss-Newton refinement layer. | Strong physical innovation point. | Train `--focal_max_scale 0.02 --focal_loss_weight 0.05`. |
| 12 | Cross-dataset domain adaptation | P1 | GRL+FiLM wrapper; unified skeleton mapping. | If H36M zero-shot < 20 mm, strong publishable story. | Small-domain-adapt experiment. |
| 13 | Real-time inference optimization | P2 | Factorized attention, SDPA/FlashAttention, distillation. | Latency/throughput numbers for paper. | Profile current PP model first. |
| 14 | Occlusion / partial visibility | P1 | Explicit visibility head + occlusion robustness evaluation. | Complete occlusion-robustness story. | Train visibility-gated model; evaluate under synthetic occlusion. |
| 15 | Self-supervised / masked pre-training | P1 | Mask ratio, view vs time masking, data-efficiency curves. | Quantify pre-training → fine-tuning data efficiency. | Same as direction 1; add data-efficiency curve. |
| 16 | Multi-person association | P2 | Extend loader + geometry association to multi-person. | System-level extension; new application scenarios. | Write `associate_multi_person_synthetic.py`. |
| 17 | Action semantics / category prior | P2 | Inject action/category embedding into PP model. | Per-action error reduction on H36M. | Build `ActionAwareDataset` + H36M experiment. |
| 18 | 3D Gaussian splatting / novel-view synthesis | P2 | Joint Gaussian rendering consistency as auxiliary regularizer. | Novel but risky; may conflict with lightweight narrative. | Isolated smoke on synthetic MPI. |
| 19 | Interpretability & failure analysis | **P0** | Per-joint/per-view failure profile; visualize weights, PP correction, residuals. | Guide calibration/visibility directions; provide paper figures. | Adapt `analyze_failures_temporal_mpiinf3dhp.py` to PP model. |
| 20 | Evaluation protocol, metrics & reproducibility | **P0** | Unified `BenchmarkProtocol`, standard splits, root-relative/velocity metrics, multi-seed. | Foundation for publishable result table. | Implement `benchmark_protocol.py` + `run_repeated_seeds.py`. |

## Top-5 P0 Actions

1. **Camera calibration robustness (principal point / extrinsic / focal / distortion)**
   - Evaluate the current curriculum checkpoint.
   - If not met, enable `--focal_max_scale 0.02 --focal_loss_weight 0.05` and stronger extrinsic perturbation.
   - Goal: clean ≤ 9.6 mm, rot_0.5° < 12 mm, focal_1% < 14 mm.

2. **Visibility-aware adaptive fusion**
   - Once the GPU is free, run `scripts/run_crossview_pp_visibility_wsl.sh`.
   - Warm-start from the best PP checkpoint with `view_dropout_rate=0.2` and `visibility_loss_weight=0.1`.
   - Goal: clean ≤ 9.6 mm, ≥ 10% relative gain at 30% occlusion.

3. **Variable-view inference & view-dropout training**
   - Generate MPJPE@k curve (k = 2..14) for the best PP model using `eval_variable_views.py`.
   - Combine with view-dropout / visibility training.
   - Goal: k=14 matches ~9.3 mm baseline; k=4–10 degrades gracefully.

4. **Interpretability & failure analysis**
   - Adapt `analyze_failures_temporal_mpiinf3dhp.py` to the `crossview_residual_pp` model.
   - Visualize per-view fusion weights, PP correction magnitude, and residual correction.
   - Produce paper-ready failure heatmaps and guidance for P1 directions.

5. **Evaluation protocol, metrics & reproducibility**
   - Implement `motionflow_mv/eval/benchmark_protocol.py`.
   - Add root-relative MPJPE, velocity MPJPE, and bone-length error to `motionflow_mv/eval/metrics.py`.
   - Fix H36M S9/S11 preprocessing; include MPI S6–S8 in standard test set.
   - Run 3–5 seeds and write `manifest.json` per run.

## Immediate non-GPU actions

1. Implement `benchmark_protocol.py` and `run_repeated_seeds.py` skeleton.
2. Run WebBridge benchmark + `audit_webbridge_npz.py` to produce cross-dataset baseline and quality report.
3. Adapt / add failure-analysis and weight-visualization scripts for the PP model.
4. Prepare code skeletons for SSL pre-training, action-aware, multi-person synthetic, and spatial pyramid (no training yet).
5. Update paper materials: robustness table, variable-view curve sketch, failure-case heatmap.

## Success criteria for next review

- Calibration robustness direction provides an updated rot/trans/focal/pp matrix.
- Visibility-gated model is trained and reports clean + occlusion robustness results.
- Variable-view curve and WebBridge cross-dataset benchmark are produced.
- Failure analysis pinpoints the main failure modes and feeds back into P1 priority adjustments.

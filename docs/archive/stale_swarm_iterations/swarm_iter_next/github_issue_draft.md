# [Swarm Iteration Next] MotionFlow-MultiView research findings and roadmap for ICRA/CVPR 2027

## Summary

This issue tracks the consolidated output of the 20-agent research swarm for the next iteration of MotionFlow-MultiView. We evaluated ~20 architecture and infrastructure directions; the current best model remains `RayAttentionFusionModelTemporalResidual` at **10.46 mm MPJPE** on MPI-INF-3DHP. The deliverables include new design reports, model prototypes, training/evaluation scripts, reproducibility harnesses, and paper-figure generators. This issue serves as the single source of truth for findings and next steps.

## Current best model

| Dataset | Metric | Value | Checkpoint / Reference |
|---|---|---|---|
| MPI-INF-3DHP S1→S2/Seq1 | MPJPE | **10.46 mm** | `outputs/ray_attention_temporal_residual_final5.pth` |
| MPI-INF-3DHP S1→S2/Seq1 | PA-MPJPE | 8.24 mm | `docs/swarm_iter7/exploration_summary.md` |
| MPI-INF-3DHP | Params | 243,428 | `d=64`, `residual_hidden=128` |
| Human3.6M S1→S5 | MPJPE | 5.71 mm | `docs/swarm_iter6/FINAL_ITERATION_REPORT.md` |
| Lightweight MPI | MPJPE | 13.22 mm | 66,420 params (`d=32`, `residual_hidden=64`) |

## Swarm deliverables (20 tasks)

### 1. Geometry and triangulation baselines
- `docs/swarm_iter_next/implement_robust_triangulation_baseline/` — IRLS/Charbonnier robust triangulation baseline as a `FusionModule`.
- `docs/swarm_iter_next/design_graph_joint_relation/` — full MPI-INF-3DHP skeleton graph joint relation design.
- `motionflow_mv/fusion/principal_point_correction.py` — principal-point correction layer (task_05).

### 2. Robustness and uncertainty
- `docs/swarm_iter_next/design_adaptive_view_selection/` — adaptive soft gate for variable view selection.
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py` — uncertainty-weighted triangulation model (task_12).
- `motionflow_mv/fusion/visibility_gated_fusion.py` — explicit visibility-gated fusion for occluded views (task_02).
- `docs/swarm_iter_next/design_uncertainty_calibration.md` — post-hoc temperature scaling for predicted variance.

### 3. Spatio-temporal and cross-view architectures
- `motionflow_mv/fusion/ray_attention_spatiotemporal_model.py` — cross-view spatio-temporal transformer (task_07).
- `docs/swarm_iter_next/design_camera_positional_encoding_report.md` — geometry-based camera positional encoding (CamPE).
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_v2_model.py` — CamPE integration (task_08).
- `docs/swarm_iter_next/design_cross_view_attention_variants/` — geometry-aware cross-view attention bias.
- `docs/swarm_iter_next/design_hierarchical_temporal_model/` — hierarchical temporal feature pyramid.

### 4. Data, training, and domain adaptation
- `motionflow_mv/data/synthetic_3d_dataset.py` improvements and `experiments/generate_synthetic_multiview_dataset.py` (task_03).
- `experiments/train_mixed_dataset.py` + `motionflow_mv/data/mixed_dataset.py` — mixed MPI+H36M+AIST training (task_06).
- `motionflow_mv/fusion/domain_adaptation_wrapper.py` — synthetic-to-real domain-adaptive wrapper (task_14).
- `experiments/train_learned_tri_v1_smoke.py` — smoke training of learned triangulation v1 (task_13).

### 5. Heads, shape, and efficiency
- `docs/swarm_iter_next/design_multi_task_shape_pose/` — multi-task shape/pose head design (task_04).
- `motionflow_mv/fusion/multi_task_shape_pose.py` — multi-task shape/pose head implementation (task_04).
- `docs/swarm_iter_next/design_self_supervised_pretext/` — self-supervised masked-view pretext task (task_20-adjacent exploration).
- `motionflow_mv/fusion/variable_view_inference.py` + `experiments/eval_variable_views.py` — variable view-count inference (task_17).
- `experiments/benchmark_runtime.py` + `docs/swarm_iter_next/runtime_benchmark_report.md` — RTX 4090 real-time benchmark (task_18).
- `docs/swarm_iter_next/design_a800_benchmark_script/` — A800 benchmark script and plan (task_15).

### 6. Integration, reproducibility, and paper artifacts
- `motionflow_mv/pipeline_multiview_plugin.py` — `MultiViewFusionPlugin` pipeline integration (task_19).
- `docs/swarm_iter_next/design_docker_reproducibility/` — Docker reproducibility and A800 benchmark scripts (task_15).
- `experiments/generate_paper_figures.py` + `docs/figures/` — paper figures and tables (task_16).
- `docs/swarm_iter_next/design_github_issue_pr_template/` — issue/PR templates and validator (task_20).

## Key findings

1. **Residual refinement on weighted DLT is the strongest component.** The temporal-only residual model outperforms cross-view and uncertainty variants in the current capacity regime.
2. **Cross-view attention is promising but needs a geometric bias.** Spatio-temporal `(time, view)` attention overfits on full data; a ray-angle bias or CamPE is required.
3. **Variable-view and cross-dataset transfer need geometry-based encodings.** Learned `view_pos_embed` is dataset-specific; CamPE removes this limitation.
4. **Robustness to occlusion is excellent; Gaussian noise is the main failure mode.** Adaptive view selection and uncertainty calibration can close this gap.
5. **Reproducibility and benchmarking need containerized, deterministic runs.** The A800 and RTX-4090 benchmark scripts provide the required harness.

## Next steps

- [ ] **Convergence run:** run the temporal-residual model to full convergence and confirm the 10.46 mm result.
- [ ] **CamPE integration:** implement `ray_attention_temporal_residual_campe_v2_model.py` and smoke-test variable-view inference.
- [ ] **Adaptive view selection:** integrate the Gumbel-softmax selector and evaluate `k ∈ {2, 3, 4}` under occlusion.
- [ ] **Cross-view geometry bias:** implement the ray-angle bias in the spatio-temporal transformer and re-evaluate.
- [ ] **Uncertainty calibration:** add `motionflow_mv/eval/calibration.py` and temperature scaling.
- [ ] **Mixed-dataset training:** run the MPI+H36M+AIST mixed trainer and report cross-dataset MPJPE.
- [ ] **Real-world GVHMR demo:** run `experiments/demo_gvhmr_multiview_projection.py` on a captured sequence.
- [ ] **Docker reproducibility:** build and validate the CUDA 12.1 container on A800.
- [ ] **Paper figures:** generate architecture, robustness, and qualitative figures.
- [ ] **Open tracking PR:** commit swarm deliverables and close this issue.

## Blockers

- **GitHub CLI / credentials:** `gh` is not authenticated, so automated issue/PR creation is blocked. Workaround: use these drafts with manual submission.
- **A800 access:** some full-training benchmarks require the A800 node; smoke tests can run on RTX 4090 / CPU.
- **Real-world demo data:** GVHMR multi-view capture is not yet available.

## Post-swarm update

After the swarm completed, a full-size camera-perturbation ablation (d=64, h=128, 10 epochs, 1 000 clips/sequence) was run on the local RTX 4090:

| Perturbation | Full with perturbation (mm) |
|---|---:|
| Clean | 14.15 |
| Rotation ±0.5° | 19.47 |
| Rotation ±1.0° | 30.22 |
| Principal point ±3 px | 1929.25 |
| Principal point ±5 px | 2132.83 |

A no-perturbation full-model baseline is currently running for a fair comparison. The principal-point failure mode remains catastrophic, so the immediate next step is to train and evaluate the new `RayAttentionFusionModelTemporalResidualPrincipalPoint` model with synthetic principal-point perturbations. Training scripts and evaluation scripts have been added; the smoke test already passes.

## Related files

- `docs/swarm_iter6/github_issue_draft.md` — previous iteration issue.
- `docs/swarm_iter6/github_pr_draft.md` — previous iteration PR.
- `docs/swarm_iter7/exploration_summary.md` — prior best results summary.
- `docs/swarm_iter_next/design_github_next_steps/report.md` — this issue/PR design rationale.
- `docs/swarm_iter_next/design_github_next_steps/validate_drafts.py` — smoke validator for these drafts.

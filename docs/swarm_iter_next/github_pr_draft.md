# [Swarm Iteration Next] Consolidated research deliverables and roadmap

## Summary

This PR adds the consolidated output of the 20-agent research swarm for the next iteration of MotionFlow-MultiView. It includes design reports, architecture prototypes, training/evaluation scripts, reproducibility harnesses, and paper-figure generators produced during the swarm. No existing model file was modified; all changes are additive.

## Key changes

### New model variants and components
- `motionflow_mv/fusion/robust_triangulation_baseline.py` + `_module.py` — IRLS/Charbonnier robust triangulation baseline as a `FusionModule`.
- `motionflow_mv/fusion/visibility_gated_fusion.py` — explicit visibility-gated fusion for occluded views.
- `motionflow_mv/fusion/multi_task_shape_pose.py` — multi-task shape/pose head.
- `motionflow_mv/fusion/principal_point_correction.py` — principal-point correction layer.
- `motionflow_mv/fusion/ray_attention_spatiotemporal_model.py` — cross-view spatio-temporal transformer.
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_v2_model.py` — CamPE integration.
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_graph_mpi_v2_model.py` — full MPI skeleton graph joint relation.
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_adaptive_softgate_v2_model.py` — improved adaptive soft gate.
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py` — uncertainty-weighted triangulation.
- `motionflow_mv/fusion/variable_view_inference.py` — variable view-count inference wrapper.
- `motionflow_mv/fusion/domain_adaptation_wrapper.py` — synthetic-to-real domain-adaptive wrapper.
- `motionflow_mv/pipeline_multiview_plugin.py` — `MultiViewFusionPlugin` pipeline integration.

### New training, evaluation, and data scripts
- `experiments/eval_robust_triangulation_baseline.py`
- `experiments/train_visibility_gated_mpiinf3dhp.py`
- `experiments/generate_synthetic_multiview_dataset.py`
- `experiments/train_multitask_mpiinf3dhp.py`
- `experiments/train_mixed_dataset.py`
- `experiments/train_spatiotemporal_mpiinf3dhp.py`
- `experiments/train_ray_attention_temporal_residual_reprojgate_full_mpiinf3dhp.py`
- `experiments/train_uncertainty_v2_mpiinf3dhp.py`
- `experiments/train_learned_tri_v1_smoke.py`
- `experiments/train_domain_adapt_mpiinf3dhp.py`
- `experiments/eval_variable_views.py`
- `experiments/benchmark_runtime.py`
- `experiments/generate_paper_figures.py`

### New design reports and infrastructure
- `docs/swarm_iter_next/implement_robust_triangulation_baseline`
- `docs/swarm_iter_next/design_adaptive_view_selection`
- `docs/swarm_iter_next/design_graph_joint_relation`
- `docs/swarm_iter_next/design_camera_positional_encoding_report.md`
- `docs/swarm_iter_next/design_self_supervised_pretext`
- `docs/swarm_iter_next/design_multi_task_shape_pose`
- `docs/swarm_iter_next/design_a800_benchmark_script`
- `docs/swarm_iter_next/design_docker_reproducibility`
- `docs/swarm_iter_next/design_github_issue_pr_template`
- `docs/swarm_iter_next/design_github_next_steps`
- `docs/swarm_iter_next/github_issue_draft.md`
- `docs/swarm_iter_next/github_pr_draft.md`

## Verified results

| Model / Artifact | Metric | Value |
|---|---|---|
| `RayAttentionFusionModelTemporalResidual` (current best) | MPI-INF-3DHP MPJPE | **10.46 mm** |
| Lightweight residual (`d=32, h=64`) | MPI-INF-3DHP MPJPE | 13.22 mm |
| `RayAttentionFusionModelTemporalResidual` | Human3.6M MPJPE | 5.71 mm |

Full benchmark tables are in `docs/swarm_iter6/FINAL_ITERATION_REPORT.md` and `docs/swarm_iter7/exploration_summary.md`.

## Testing

Run the smoke validator for the new drafts and templates:

```bash
conda run -n mf python docs/swarm_iter_next/design_github_next_steps/validate_drafts.py
```

Run the existing ray-attention test suite to confirm no regressions:

```bash
conda run -n mf pytest tests/ -v
```

Smoke-test the A800 benchmark prototype (CPU-friendly with `--smoke`):

```bash
python docs/swarm_iter_next/design_a800_benchmark_script/benchmark_a800_prototype.py --smoke
```

## Checklist

- [x] All 20 swarm deliverables documented under `docs/swarm_iter_next/`.
- [x] New code files are additive and do not modify `ray_attention_temporal_residual_model.py`.
- [x] Smoke tests pass (`pytest tests/ -v`).
- [x] Issue/PR drafts and validator added.
- [ ] Full A800 convergence run (follow-up issue).
- [ ] Real-world GVHMR demo (follow-up issue).
- [ ] `gh` CLI authentication for automated issue/PR creation.

## Related issues

Closes the umbrella issue `docs/swarm_iter_next/github_issue_draft.md`.

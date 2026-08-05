# Swarm Iteration Summary — 2026-08-06

**Iteration focus:** Integration, evaluation protocol, reproducibility, and non-GPU tooling for the 20-agent swarm synthesis.
**Current best:** MPI-INF-3DHP clean **9.32 mm** / PA-MPJPE **5.37 mm** (cross-view residual + principal point, d=64/h=128, 20 epochs).

## Summary of changes

This iteration consolidates the cross-view residual + principal-point (PP) model line, adds evaluation/benchmark scaffolding, and prepares the next round of P0 experiments (visibility gating, variable-view inference, calibration curriculum, reproducibility).

- Hardened the best cross-view PP model and wrapped it for plug-in use.
- Added variable-view evaluation (`eval_variable_views.py`) and a plotting harness for MPJPE@k curves.
- Added the WebBridge benchmark harness (`run_webbridge_benchmark.py`, `summarize_webbridge_benchmark.py`) for cross-dataset testing.
- Extended evaluation metrics with root-relative MPJPE, velocity MPJPE, and bone-length error.
- Added multi-seed repeated training harness (`run_repeated_seeds.py`).
- Added self-supervised masked-view pre-training skeleton (`ssl_dataset.py`, `pretrain_ray_attention_ssl.py`).
- Added failure-analysis / interpretability script for the cross-view PP model.
- Added a large set of WSL runner scripts to queue/launch the above experiments.

## New files

### Core model modules (`motionflow_mv/fusion/`)
- `ray_attention_temporal_crossview_residual_principal_point_model.py` — best PP model
- `ray_attention_temporal_crossview_residual_principal_point_refined_model.py` — two-stage refined PP variant (negative result)
- `ray_attention_temporal_crossview_residual_principal_point_visibility_model.py` — visibility-gated PP model
- `ray_attention_temporal_crossview_residual_principal_point_module.py` — reusable module wrapper
- `ray_attention_temporal_crossview_residual_campe_v2_model.py` — CamPE v2 cross-view residual
- `ray_attention_temporal_crossview_residual_adaptive_view_selection_model.py` — adaptive view selection
- `ray_attention_temporal_residual_principal_point_model.py`
- `variable_view_inference.py`
- `visibility_gated_fusion.py`
- Other exploratory variants: `ray_attention_v{2,3,4}_model.py`, `ray_attention_temporal_*_model.py`, `ray_attention_spatiotemporal_model.py`, `robust_triangulation_baseline*.py`, etc.

### Data & losses
- `motionflow_mv/data/ssl_dataset.py` — masked-view SSL pre-training dataset
- `motionflow_mv/losses/bone_length.py` — bone-length consistency loss

### Evaluation & benchmark scripts (`experiments/`)
- `eval_variable_views.py` — variable-view MPJPE@k evaluation
- `plot_variable_views.py` — plot MPJPE@k curves
- `run_webbridge_benchmark.py` — WebBridge benchmark runner
- `summarize_webbridge_benchmark.py` — summarize WebBridge results
- `analyze_failures_crossview_pp.py` — failure analysis / interpretability for PP model
- `run_repeated_seeds.py` — multi-seed repeated training harness
- `eval_full_metrics.py` changes now support cross-view PP, variable-view, visibility-gated, and mixed-dataset models

### Training scripts (`experiments/`)
- `train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` — best PP model trainer
- `train_ray_attention_temporal_crossview_residual_principal_point_visibility_mpiinf3dhp.py` — visibility-gated trainer
- `train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py`
- `train_ray_attention_temporal_residual_mpiinf3dhp.py`
- `pretrain_ray_attention_ssl.py` — SSL pre-training skeleton

### WSL launchers (`scripts/`)
- `run_crossview_pp_curriculum_wsl.sh`
- `run_crossview_pp_visibility_wsl.sh`
- `run_crossview_pp_refined_small_wsl.sh`
- `run_crossview_pp_h36m_full_ppw005_wsl.sh`
- `run_crossview_pp_full_ppw005_20ep_wsl.sh`
- `run_crossview_campe_v2_small_wsl.sh` / `run_crossview_campe_v2_full_wsl.sh`
- `run_crossview_adaptive_small_wsl.sh`
- `eval_crossview_pp_curriculum_wsl.sh`
- `eval_variable_views_crossview_pp_wsl.sh`
- `eval_pp_supervised_small_wsl.sh` / `eval_pp_supervised_full_wsl.sh`
- `eval_crossview_adaptive_small_wsl.sh`
- `eval_crossview_campe_v2_small_wsl.sh`
- `run_webbridge_benchmark_wsl.sh`
- `run_ssl_pretrain_wsl.sh` / `run_ssl_pretrain_h36m_wsl.sh`
- `analyze_failures_crossview_pp_wsl.sh`
- `run_focal_small_highperturb_wsl.sh`
- `push_to_github.sh`

### Configs
- `configs/benchmark_webbridge_crossview_residual_smoke.yaml`
- `configs/benchmark_webbridge_mpi_smoke.yaml`
- `configs/train_ray_attention_reproducible.yaml`

### Tests (`tests/`)
- `test_ray_attention_crossview_residual_principal_point_visibility.py`
- `test_principal_point_correction.py`
- `test_ray_attention_spatiotemporal.py`
- `test_multiscale_temporal.py`
- `test_multiview_adapter.py`
- `test_pipeline_multiview_plugin*.py`
- `test_ray_attention_temporal*.py`
- `test_synthetic_amass_augmentation.py`
- `test_ray_attention_v4_residual.py`
- `test_ray_attention_temporal_uncertainty.py`

### Documentation & results
- `docs/results_icra_cvpr_2027.md` — consolidated results table
- `docs/experiment_log_icra_cvpr_2027.md` — chronological experiment log
- `docs/next_iteration_plan_swarm.md` — 20-agent swarm next-iteration plan
- `docs/paper_draft_icra_cvpr_2027.md`
- `docs/figures/icra2027/*.png` — paper figures (bar charts, per-joint MPJPE, robustness grid)
- `docs/results_cross_dataset.md`
- `docs/results_h36m_v1.md`, `docs/results_h36m_v1_metric.md`, `docs/results_h36m_v2.md`
- `docs/swarm_iter_next/` — design documents and demos for next directions

## Modified files

- `motionflow_mv/eval/metrics.py` — added root-relative MPJPE, velocity MPJPE, bone-length error
- `experiments/eval_full_metrics.py` — support cross-view PP, variable-view, visibility-gated, and mixed-dataset evaluation; optional parent skeleton for metrics
- `experiments/eval_variable_views.py` — added `--output_json`, `--output_csv`; support for multi-output and sampled view subsets
- `experiments/run_webbridge_benchmark.py` — variable-view inference support
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` — calibration perturbation curriculum, warm-start option
- `motionflow_mv/fusion/triangulation.py` — variable-view / robust triangulation changes
- `motionflow_mv/data/synthetic_3d_dataset.py` — view-mask broadcast fix
- `motionflow_mv/losses/__init__.py` — export bone-length loss
- `motionflow_mv/fusion/__init__.py` — register new models
- `docs/results_icra_cvpr_2027.md` — updated with latest numbers
- `docs/experiment_log_icra_cvpr_2027.md` — log updated with 20-agent swarm synthesis
- `docs/next_iteration_plan_swarm.md` — updated plan
- `docs/paper_draft_icra_cvpr_2027.md` — paper draft updates
- `configs/benchmark_webbridge_crossview_residual_smoke.yaml` — variable-view support

## Uncommitted changes

- `experiments/eval_variable_views.py` — adds `--output_csv` to save per-k MPJPE summary as CSV (on top of existing `--output_json`).

## Negative results logged

- Two-stage refined PP correction: 14.53 mm clean (dropped).
- CamPE v2 small: 14.39 mm clean (dropped).
- CamPE v2 full: 10.53 mm val (dropped).
- Mixed MPI+H36M small: MPI 11.64 mm but H36M 101 mm (poor cross-dataset generalization).

## GitHub status update

- `gh issue comment 21` — **not executed**: `gh` CLI is not installed in this environment (`command not found`).
- `gh pr comment 17` — **not executed**: `gh` CLI is not installed in this environment (`command not found`).

Please run the GitHub comments manually from an environment with `gh` authenticated, or provide an alternate method.

## Suggested commit message

```
swarm(iter): 2026-08-06 integration, eval protocol and reproducibility tooling

- Harden cross-view residual + principal-point model (best 9.32 mm clean / 5.37 mm PA).
- Add variable-view MPJPE@k evaluation and plotting harness.
- Add WebBridge benchmark harness and summary script.
- Extend metrics: root-relative, velocity, and bone-length error.
- Add multi-seed repeated training harness and SSL pre-training skeleton.
- Add failure-analysis/interpretability script for cross-view PP model.
- Add WSL runner scripts for curriculum, visibility, WebBridge, SSL, and eval tasks.
- Log negative results (refined PP, CamPE v2, mixed-dataset H36M).

Refs: docs/next_iteration_plan_swarm.md
```

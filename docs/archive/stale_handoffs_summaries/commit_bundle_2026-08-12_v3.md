# Commit Bundle — 2026-08-12 v3 (Final)

> Generated: 2026-08-12T07:05:31Z
> Branch: `main`
> HEAD: `89d794b` — *docs+infra: final H36M true-GT / AIST++ docs, configs, and scripts for CVPR 2027*
> Note: Local `main` is **1 commit ahead** of `origin/main` and has not been pushed.

## Bundle Purpose

This is the **v3 / final** staging bundle for the v57 H36M true-GT medium completion, the v25 true-GT ablation suite, and the surrounding CVPR 2027 documentation and infrastructure updates. It supersedes `docs/commit_bundle_2026-08-12_v2.md` and captures every new/modified doc, config, script, and experiment helper that should enter the repository, while continuing to exclude transient artifacts and large binaries.

### Scope

- Final v57 H36M true-GT medium results, handoff, and leaderboard updates.
- v25 true-GT ablation suite: baseline fix, geometry regularization, and mixed-dataset ablations.
- MPI real-detected-2D generation pipeline (CPU-only) and split definitions.
- SOTA baseline runner scripts (VoxelPose, MVPose) for H36M true-GT.
- Variable-views eval scripts and AIST++ medium-run queue helpers for v57/v80.
- Supporting planning/checklist docs (48-hour plan, citation verification, submission checklist, paper results table, open blockers).

It intentionally **excludes** transient cleanup artifacts, large MediaPipe binary task files, older commit-bundle drafts, and the transient `git_status_summary.md` snapshot.

## Suggested Commit Message

```text
docs+scripts: v57 H36M true-GT finish, v25 ablation suite, and CVPR 2027 handoff updates

- Record final v57 H36M true-GT medium results in AGENTS.md, README.md,
  docs/cvpr2027_status.md, docs/handoff_qwen3.8max.md, docs/results_true_gt_h36m.md,
  docs/roadmap_qwen3.8max.md, and docs/project_status_onepager.md.
- Add the v25 true-GT ablation suite:
  configs/ablations/v25_true_gt_baseline_fix.yaml,
  configs/ablations/v25_true_gt_geometry_regularization.yaml,
  configs/ablations/v25_true_gt_mixed_dataset.yaml,
  scripts/run_v25_ablation_true_gt_baseline.sh,
  scripts/run_v25_ablation_geometry_regularization.sh,
  scripts/run_v25_ablation_mixed_dataset.sh,
  scripts/run_v25_ablations_sequential.sh,
  docs/v25_ablation_plan.md, docs/ablation_schedule.md,
  docs/v25_v80_failure_analysis.md.
- Update configs/splits/mix_true_gt_v2.yaml and add balanced mix generator
  scripts/generate_mix_true_gt_v2_balanced.py plus validator
  scripts/validate_mix_true_gt_v2.py.
- Update scripts/generate_mpi_detected_2d_from_avi.py and add the MPI-INF-3DHP
  detected-2D smoke split configs/splits/mpi_inf_3dhp_detected_2d_smoke.yaml
  for CPU-only real detected 2D generation.
- Add SOTA baseline eval scripts for H36M true-GT:
  scripts/run_voxelpose_h36m_true_gt.sh,
  scripts/run_mvpose_h36m_true_gt.sh.
- Add variable-views eval and queue scripts for the ongoing/queued medium runs:
  scripts/eval_variable_views_v57_h36m_true_gt_medium.sh,
  scripts/eval_variable_views_v80_h36m_true_gt_medium.sh,
  scripts/wait_v57_then_v80_aistpp_train_val.sh,
  scripts/run_aistpp_medium_queue.sh.
- Update paper draft, roadmap, and status docs:
  docs/paper_draft_icra_cvpr_2027.md, docs/cvpr2027_status.md,
  docs/cvpr2027_submission_checklist.md, docs/citation_verification.md,
  docs/paper_results_table.md, docs/48h_plan_2026-08-12.md,
  docs/open_blockers.md, docs/sparse_view_eval_protocol.md,
  docs/session_summary_2026-08-11.md, docs/leaderboard_summary_2026-08-11.md,
  docs/next_turn_for_qwen3.8max.md, docs/handoff_qwen3.8max_session_summary.md,
  docs/qwen3.8max_24h_plan.md, docs/v80_checkpoint_validation.md,
  docs/3dpw_audit.md, docs/a800_missing_data_check.md,
  docs/gpu_contention_check.md.
- Update scripts/sota_baselines/README.md with current true-GT protocol notes.

Does NOT include:
- archive_cleanup_20260811/ and cleanup_safe_20260811.sh (cleanup artifacts)
- models/mediapipe/ (large binary task files)
- tmp_mpi_debug_f0v0.png (transient debug image)
- docs/commit_bundle_2026-08-11.md, docs/commit_bundle_2026-08-11_v2.md,
  docs/commit_bundle_2026-08-12.md, docs/commit_bundle_2026-08-12_v2.md,
  docs/commit_bundle_2026-08-12_v3.md, docs/git_status_summary.md
  (transient bundle/status drafts)
```

## Files Included

### Documentation updates (modified)

- `AGENTS.md`
- `README.md`
- `docs/cvpr2027_status.md`
- `docs/handoff_qwen3.8max.md`
- `docs/paper_draft_icra_cvpr_2027.md`
- `docs/results_mpi_detected_dlt.md`
- `docs/results_true_gt_h36m.md`
- `docs/roadmap_qwen3.8max.md`
- `scripts/sota_baselines/README.md`

### New documentation

- `docs/3dpw_audit.md`
- `docs/48h_plan_2026-08-12.md`
- `docs/a800_missing_data_check.md`
- `docs/ablation_schedule.md`
- `docs/citation_verification.md`
- `docs/commit_bundle_2026-08-12.md`
- `docs/commit_bundle_2026-08-12_v2.md`
- `docs/cvpr2027_pivot_for_new_collaborators.md`
- `docs/cvpr2027_submission_checklist.md`
- `docs/gpu_contention_check.md`
- `docs/handoff_qwen3.8max_session_summary.md`
- `docs/leaderboard_summary_2026-08-11.md`
- `docs/next_turn_for_qwen3.8max.md`
- `docs/open_blockers.md`
- `docs/paper_results_table.md`
- `docs/project_status_onepager.md`
- `docs/qwen3.8max_24h_plan.md`
- `docs/session_summary_2026-08-11.md`
- `docs/sparse_view_eval_protocol.md`
- `docs/v25_ablation_plan.md`
- `docs/v25_v80_failure_analysis.md`
- `docs/v57_ablation_analysis.md`
- `docs/v57_checkpoint_validation.md`
- `docs/v80_checkpoint_validation.md`

### Configs

- `configs/splits/mix_true_gt_v2.yaml` (modified)
- `configs/ablations/v25_true_gt_baseline_fix.yaml`
- `configs/ablations/v25_true_gt_geometry_regularization.yaml`
- `configs/ablations/v25_true_gt_mixed_dataset.yaml`
- `configs/splits/mpi_inf_3dhp_detected_2d_smoke.yaml`

### Scripts / experiments

- `scripts/generate_mpi_detected_2d_from_avi.py` (modified)
- `scripts/generate_mix_true_gt_v2_balanced.py`
- `scripts/validate_mix_true_gt_v2.py`
- `scripts/eval_variable_views_v57_h36m_true_gt_medium.sh`
- `scripts/eval_variable_views_v80_h36m_true_gt_medium.sh`
- `scripts/run_mvpose_h36m_true_gt.sh`
- `scripts/run_v25_ablation_geometry_regularization.sh`
- `scripts/run_v25_ablation_mixed_dataset.sh`
- `scripts/run_v25_ablation_true_gt_baseline.sh`
- `scripts/run_v25_ablations_sequential.sh`
- `scripts/run_voxelpose_h36m_true_gt.sh`
- `scripts/run_aistpp_medium_queue.sh`
- `scripts/wait_v57_then_v80_aistpp_train_val.sh`

## Commands to Stage and Commit

```bash
# 1. Stage modified docs / README / AGENTS
 git add AGENTS.md README.md
 git add docs/cvpr2027_status.md docs/handoff_qwen3.8max.md
 git add docs/paper_draft_icra_cvpr_2027.md
 git add docs/results_mpi_detected_dlt.md docs/results_true_gt_h36m.md
 git add docs/roadmap_qwen3.8max.md
 git add scripts/sota_baselines/README.md

# 2. Stage new documentation
 git add docs/3dpw_audit.md docs/48h_plan_2026-08-12.md
 git add docs/a800_missing_data_check.md docs/ablation_schedule.md
 git add docs/citation_verification.md
 git add docs/commit_bundle_2026-08-12.md
 git add docs/commit_bundle_2026-08-12_v2.md
 git add docs/cvpr2027_pivot_for_new_collaborators.md
 git add docs/cvpr2027_submission_checklist.md
 git add docs/gpu_contention_check.md
 git add docs/handoff_qwen3.8max_session_summary.md
 git add docs/leaderboard_summary_2026-08-11.md
 git add docs/next_turn_for_qwen3.8max.md
 git add docs/open_blockers.md docs/paper_results_table.md
 git add docs/project_status_onepager.md
 git add docs/qwen3.8max_24h_plan.md docs/session_summary_2026-08-11.md
 git add docs/sparse_view_eval_protocol.md
 git add docs/v25_ablation_plan.md
 git add docs/v25_v80_failure_analysis.md
 git add docs/v57_ablation_analysis.md
 git add docs/v57_checkpoint_validation.md
 git add docs/v80_checkpoint_validation.md

# 3. Stage configs
 git add configs/splits/mix_true_gt_v2.yaml
 git add configs/ablations/v25_true_gt_baseline_fix.yaml
 git add configs/ablations/v25_true_gt_geometry_regularization.yaml
 git add configs/ablations/v25_true_gt_mixed_dataset.yaml
 git add configs/splits/mpi_inf_3dhp_detected_2d_smoke.yaml

# 4. Stage scripts
 git add scripts/generate_mpi_detected_2d_from_avi.py
 git add scripts/generate_mix_true_gt_v2_balanced.py
 git add scripts/validate_mix_true_gt_v2.py
 git add scripts/eval_variable_views_v57_h36m_true_gt_medium.sh
 git add scripts/eval_variable_views_v80_h36m_true_gt_medium.sh
 git add scripts/run_mvpose_h36m_true_gt.sh
 git add scripts/run_v25_ablation_geometry_regularization.sh
 git add scripts/run_v25_ablation_mixed_dataset.sh
 git add scripts/run_v25_ablation_true_gt_baseline.sh
 git add scripts/run_v25_ablations_sequential.sh
 git add scripts/run_voxelpose_h36m_true_gt.sh
 git add scripts/run_aistpp_medium_queue.sh
 git add scripts/wait_v57_then_v80_aistpp_train_val.sh

# 5. Review staged changes
 git status --short
 git diff --cached --stat

# 6. Commit with the message above
 git commit -F- <<'EOF'
docs+scripts: v57 H36M true-GT finish, v25 ablation suite, and CVPR 2027 handoff updates

- Record final v57 H36M true-GT medium results in AGENTS.md, README.md,
  docs/cvpr2027_status.md, docs/handoff_qwen3.8max.md, docs/results_true_gt_h36m.md,
  docs/roadmap_qwen3.8max.md, and docs/project_status_onepager.md.
- Add the v25 true-GT ablation suite:
  configs/ablations/v25_true_gt_baseline_fix.yaml,
  configs/ablations/v25_true_gt_geometry_regularization.yaml,
  configs/ablations/v25_true_gt_mixed_dataset.yaml,
  scripts/run_v25_ablation_true_gt_baseline.sh,
  scripts/run_v25_ablation_geometry_regularization.sh,
  scripts/run_v25_ablation_mixed_dataset.sh,
  scripts/run_v25_ablations_sequential.sh,
  docs/v25_ablation_plan.md, docs/ablation_schedule.md,
  docs/v25_v80_failure_analysis.md.
- Update configs/splits/mix_true_gt_v2.yaml and add balanced mix generator
  scripts/generate_mix_true_gt_v2_balanced.py plus validator
  scripts/validate_mix_true_gt_v2.py.
- Update scripts/generate_mpi_detected_2d_from_avi.py and add the MPI-INF-3DHP
  detected-2D smoke split configs/splits/mpi_inf_3dhp_detected_2d_smoke.yaml
  for CPU-only real detected 2D generation.
- Add SOTA baseline eval scripts for H36M true-GT:
  scripts/run_voxelpose_h36m_true_gt.sh,
  scripts/run_mvpose_h36m_true_gt.sh.
- Add variable-views eval and queue scripts for the ongoing/queued medium runs:
  scripts/eval_variable_views_v57_h36m_true_gt_medium.sh,
  scripts/eval_variable_views_v80_h36m_true_gt_medium.sh,
  scripts/wait_v57_then_v80_aistpp_train_val.sh,
  scripts/run_aistpp_medium_queue.sh.
- Update paper draft, roadmap, and status docs:
  docs/paper_draft_icra_cvpr_2027.md, docs/cvpr2027_status.md,
  docs/cvpr2027_submission_checklist.md, docs/citation_verification.md,
  docs/paper_results_table.md, docs/48h_plan_2026-08-12.md,
  docs/open_blockers.md, docs/sparse_view_eval_protocol.md,
  docs/session_summary_2026-08-11.md, docs/leaderboard_summary_2026-08-11.md,
  docs/next_turn_for_qwen3.8max.md, docs/handoff_qwen3.8max_session_summary.md,
  docs/qwen3.8max_24h_plan.md, docs/v80_checkpoint_validation.md,
  docs/3dpw_audit.md, docs/a800_missing_data_check.md,
  docs/gpu_contention_check.md.
- Update scripts/sota_baselines/README.md with current true-GT protocol notes.

Does NOT include:
- archive_cleanup_20260811/ and cleanup_safe_20260811.sh (cleanup artifacts)
- models/mediapipe/ (large binary task files)
- tmp_mpi_debug_f0v0.png (transient debug image)
- docs/commit_bundle_2026-08-11.md, docs/commit_bundle_2026-08-11_v2.md,
  docs/commit_bundle_2026-08-12.md, docs/commit_bundle_2026-08-12_v2.md,
  docs/commit_bundle_2026-08-12_v3.md, docs/git_status_summary.md
  (transient bundle/status drafts)
EOF
```

## Excluded (Do Not Add)

| Path | Reason |
|------|--------|
| `archive_cleanup_20260811/` | Local archive of cleaned-up outputs; do not commit. |
| `cleanup_safe_20260811.sh` | One-off cleanup helper; not part of source tree. |
| `models/mediapipe/` | Large binary MediaPipe task files. |
| `tmp_mpi_debug_f0v0.png` | Transient debug visualization. |
| `docs/commit_bundle_2026-08-11.md` | Older proposed bundle; do not commit. |
| `docs/commit_bundle_2026-08-11_v2.md` | Older proposed bundle; do not commit. |
| `docs/commit_bundle_2026-08-12.md` | Superseded proposed bundle; do not commit. |
| `docs/commit_bundle_2026-08-12_v2.md` | Superseded proposed bundle; do not commit. |
| `docs/commit_bundle_2026-08-12_v3.md` | This proposed bundle; do not commit. |
| `docs/git_status_summary.md` | Transient status snapshot; do not commit. |

## Verification Checklist

- [ ] `git status --short` shows only the files listed under "Files Included".
- [ ] No `models/mediapipe/`, `archive_cleanup_20260811/`, `cleanup_safe_20260811.sh`, or `tmp_mpi_debug_f0v0.png` is staged.
- [ ] No `docs/commit_bundle_*.md` or `docs/git_status_summary.md` is staged.
- [ ] `git diff --cached --stat` looks reasonable (mostly docs/configs/scripts).
- [ ] Commit message body fits within 72 columns per line.
- [ ] No GPU task was launched by this commit operation.
- [ ] Push only after reviewing the local `main` commit (`89d794b`) and the new commit together.

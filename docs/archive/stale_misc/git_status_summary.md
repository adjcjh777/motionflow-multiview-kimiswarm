# Git Status Summary

> Generated: 2026-08-11T05:49:55Z
> Branch: `main`
> HEAD: `89d794b` — *docs+infra: final H36M true-GT / AIST++ docs, configs, and scripts for CVPR 2027*
> Note: Local `main` is **1 commit ahead** of `origin/main` and has not been pushed.

## 1. Repository State

- **Working tree is NOT clean.** There are unstaged changes and untracked files, but **no staged changes**.
- The most recent local commit is `89d794b`; everything listed below is uncommitted.
- **Unstaged changes:** 11 files modified in the working tree.
- **Untracked files:** 46 untracked files/folders (includes transient cleanup artifacts and large model binaries).
- No commit was performed per safety policy; run `git status --short` for the live view.

## 2. Unstaged Changes (Working Tree Only)

| File | Notes |
|------|-------|
| `AGENTS.md` | Project status / GPU occupancy updates |
| `README.md` | README additions |
| `configs/splits/mix_true_gt_v2.yaml` | Mixed true-GT v2 split updates |
| `docs/cvpr2027_status.md` | CVPR 2027 status / leaderboard update |
| `docs/handoff_qwen3.8max.md` | qwen3.8max handoff update |
| `docs/paper_draft_icra_cvpr_2027.md` | Paper draft update |
| `docs/results_mpi_detected_dlt.md` | MPI real detected 2D DLT results update |
| `docs/results_true_gt_h36m.md` | H36M true-GT leaderboard update |
| `docs/roadmap_qwen3.8max.md` | Roadmap update |
| `scripts/generate_mpi_detected_2d_from_avi.py` | MPI real detected 2D generation pipeline update |
| `scripts/sota_baselines/README.md` | SOTA baseline README update |

## 3. Untracked Files

### Documentation

| Path | Notes |
|------|-------|
| `docs/3dpw_audit.md` | 3DPW dataset audit notes |
| `docs/48h_plan_2026-08-12.md` | 48-hour plan for 2026-08-12 |
| `docs/a800_missing_data_check.md` | A800-D missing-data check notes |
| `docs/ablation_schedule.md` | Planned ablations schedule |
| `docs/citation_verification.md` | Citation verification notes |
| `docs/commit_bundle_2026-08-11.md` | Proposed commit bundle (older; do not add to commit) |
| `docs/commit_bundle_2026-08-11_v2.md` | Proposed commit bundle v2 (older; do not add to commit) |
| `docs/commit_bundle_2026-08-12.md` | Proposed commit bundle for today |
| `docs/cvpr2027_pivot_for_new_collaborators.md` | CVPR 2027 pivot summary for collaborators |
| `docs/cvpr2027_submission_checklist.md` | CVPR 2027 submission checklist |
| `docs/git_status_summary.md` | This file (transient snapshot; do not commit as source of truth) |
| `docs/gpu_contention_check.md` | GPU contention check notes |
| `docs/handoff_qwen3.8max_session_summary.md` | qwen3.8max session summary handoff |
| `docs/leaderboard_summary_2026-08-11.md` | Leaderboard snapshot |
| `docs/next_turn_for_qwen3.8max.md` | Next-turn notes |
| `docs/open_blockers.md` | Open blockers list |
| `docs/paper_results_table.md` | Paper results table |
| `docs/project_status_onepager.md` | Project status one-pager |
| `docs/qwen3.8max_24h_plan.md` | 24-hour plan for qwen3.8max |
| `docs/session_summary_2026-08-11.md` | Session summary |
| `docs/sparse_view_eval_protocol.md` | Sparse-view evaluation protocol |
| `docs/v25_ablation_plan.md` | v25 ablation plan |
| `docs/v25_v80_failure_analysis.md` | v25 / v80 failure analysis |
| `docs/v80_checkpoint_validation.md` | v80 checkpoint validation notes |

### Configs

| Path | Notes |
|------|-------|
| `configs/ablations/v25_true_gt_baseline_fix.yaml` | v25 true-GT baseline-fix ablation |
| `configs/ablations/v25_true_gt_geometry_regularization.yaml` | v25 geometry-regularization ablation |
| `configs/ablations/v25_true_gt_mixed_dataset.yaml` | v25 mixed-dataset ablation |
| `configs/splits/mpi_inf_3dhp_detected_2d_smoke.yaml` | MPI-INF-3DHP detected-2D smoke split |

### Scripts

| Path | Notes |
|------|-------|
| `scripts/eval_variable_views_v57_h36m_true_gt_medium.sh` | v57 H36M true-GT variable-views eval |
| `scripts/eval_variable_views_v80_h36m_true_gt_medium.sh` | v80 H36M true-GT variable-views eval |
| `scripts/generate_mix_true_gt_v2_balanced.py` | Balanced mix_true_gt_v2 generator |
| `scripts/run_aistpp_medium_queue.sh` | AIST++ medium queue runner |
| `scripts/run_mvpose_h36m_true_gt.sh` | MVPose H36M true-GT runner |
| `scripts/run_v25_ablation_geometry_regularization.sh` | v25 geometry-regularization ablation runner |
| `scripts/run_v25_ablation_mixed_dataset.sh` | v25 mixed-dataset ablation runner |
| `scripts/run_v25_ablation_true_gt_baseline.sh` | v25 true-GT baseline ablation runner |
| `scripts/run_v25_ablations_sequential.sh` | v25 sequential ablations runner |
| `scripts/run_voxelpose_h36m_true_gt.sh` | VoxelPose H36M true-GT runner |
| `scripts/validate_mix_true_gt_v2.py` | mix_true_gt_v2 split validator |
| `scripts/wait_v57_then_v80_aistpp_train_val.sh` | Wait for v57 then launch v80 AIST++ train/val |

### Transient / Excluded

| Path | Reason |
|------|--------|
| `archive_cleanup_20260811/` | Local cleanup archive; do not commit |
| `cleanup_safe_20260811.sh` | One-off cleanup helper; not part of source tree |
| `models/mediapipe/` | Large binary MediaPipe task files |
| `tmp_mpi_debug_f0v0.png` | Transient debug image |

## 4. Active Background Work (Do Not Duplicate)

- Per `AGENTS.md`, the local RTX 4090 may still be running the **v57 H36M true-GT medium** training run. No new GPU training or eval task was launched by this update.
- Check `nvidia-smi` / `tmux` / relevant log files before starting any GPU work.
- **A800-D `/mnt/nvme0n1/zhangzy/projects` and the A800 Docker `motionflow` service are read-only.** No writes, starts, or modifications were made there.

## 5. Key Blockers / Next Steps

1. **Commit decision:** 11 modified files and ~30 new docs/configs/scripts are ready for review. Transient artifacts should stay excluded.
2. **GPU status:** Confirm the local RTX 4090 is idle before scheduling new training/eval.
3. **Push decision:** The local `main` branch is 1 commit ahead of `origin/main`; decide whether to push after committing.
4. **Branch cleanup audit:** Completed in `docs/github_branch_cleanup_audit.md`. Local branches reduced to 15; 90 remote-only candidates remain for review; 9 stale branches are blocked by worktrees.

---

*This file was auto-generated from `git status` output. Do not commit it as the source of truth; treat it as a transient snapshot.*

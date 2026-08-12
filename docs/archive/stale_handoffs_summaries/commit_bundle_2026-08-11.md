# Commit Bundle — 2026-08-11

> Generated: 2026-08-11T11:51:10+08:00
> Branch: `main`
> HEAD: `3f81edf` — *Extend view-dropout ablation note with Shelf/Campus confirmation*

## Bundle Purpose

This bundle captures the documentation, configs, and scripts produced by the recent `qwen3.8max` handoff and the H36M true-GT / AIST++ integration work. It groups logically related changes so they can be committed together without mixing transient artifacts (MediaPipe task files, tmp images, cleanup archives) into the repository history.

## Suggested Commit Message

```text
docs+infra: sync CVPR 2027 status, H36M true-GT results, and AIST++ smoke scripts

- Update AGENTS.md / CLAUDE.md with H36M true-GT availability, in-flight
  agents (agent-51, agent-67), and read-only A800-D rules.
- Refresh handoff and paper draft docs to remove unverifiable citations.
- Add H36M true-GT and AIST++ results/docs: results_true_gt_h36m.md,
  results_true_gt_shelf_campus.md, aistpp_smoke_diagnosis.md,
  results_aistpp_dlt_baseline.md, cvpr2027_status.md, v25_training_health.md,
  data_audit_summary_2026-08-11.md, roadmap_qwen3.8max.md.
- Add AIST++/H36M true-GT smoke configs and split files.
- Add evaluation / training scripts for v25/v57/v80 on H36M true-GT and
  AIST++, plus Iskakov baseline support and SOTA baseline helpers.
- Update A800 data inventory and Docker status docs.
- Record repository cleanup execution logs.

Does NOT include:
- models/mediapipe/ (large binary task files)
- tmp_mpi_debug_f0v0.png (transient debug image)
- archive_cleanup_20260811/ and cleanup_safe_20260811.sh (local cleanup artifacts)
- docs/git_status_summary.md (transient status snapshot)
```

## Files Included

### Documentation updates

- `.github/ISSUE_TEMPLATE/v25_geometry_fusion_round.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/a800_data_inventory.md`
- `docs/a800_docker_status.md`
- `docs/aistpp_smoke_diagnosis.md`
- `docs/cleanup_executed.md`
- `docs/cleanup_plan.md`
- `docs/cvpr2027_status.md`
- `docs/data_audit_summary_2026-08-11.md`
- `docs/github_branch_cleanup.md`
- `docs/github_branch_cleanup_executed.md`
- `docs/handoff_qwen3.8max.md`
- `docs/paper_draft_icra_cvpr_2027.md`
- `docs/results_aistpp_dlt_baseline.md`
- `docs/results_true_gt_h36m.md`
- `docs/results_true_gt_shelf_campus.md`
- `docs/roadmap_qwen3.8max.md`
- `docs/v25_training_health.md`

### New configs

- `configs/benchmark_v57_h36m_true_gt_medium.yaml`
- `configs/splits/aist_only_smoke.yaml`
- `configs/splits/aistpp_train_val.yaml`
- `configs/splits/h36m_true_gt_aist_mixed_smoke.yaml`
- `configs/splits/mix_h36m_aist_shelf.yaml`

### Scripts / experiments

- `experiments/diagnose_aistpp_smoke.py`
- `experiments/run_aistpp_dlt_baseline.py`
- `experiments/train_iskakov_baseline_shelf_campus.py`
- `scripts/eval_variable_views_h36m_true_gt/`
- `scripts/fetch_mpi_real_2d.sh`
- `scripts/gpu_queue.sh`
- `scripts/run_iskakov_aist_only_smoke_local_4090.sh`
- `scripts/run_v25_aist_only_smoke_local_4090.sh`
- `scripts/run_v57_h36m_true_gt_medium.sh`
- `scripts/run_v80_aist_only_smoke_local_4090.sh`
- `scripts/run_v80_h36m_true_gt_medium.sh`
- `scripts/run_v80_h36m_true_gt_reg_v4_a800.sh`
- `scripts/sota_baselines/`

## Commands to Stage and Commit

```bash
# 1. Stage documentation updates
git add .github/ISSUE_TEMPLATE/v25_geometry_fusion_round.md
git add AGENTS.md CLAUDE.md
git add docs/a800_data_inventory.md docs/a800_docker_status.md
git add docs/aistpp_smoke_diagnosis.md
git add docs/cleanup_executed.md docs/cleanup_plan.md
git add docs/cvpr2027_status.md docs/data_audit_summary_2026-08-11.md
git add docs/github_branch_cleanup.md docs/github_branch_cleanup_executed.md
git add docs/handoff_qwen3.8max.md
git add docs/paper_draft_icra_cvpr_2027.md
git add docs/results_aistpp_dlt_baseline.md
git add docs/results_true_gt_h36m.md docs/results_true_gt_shelf_campus.md
git add docs/roadmap_qwen3.8max.md docs/v25_training_health.md

# 2. Stage new configs
git add configs/benchmark_v57_h36m_true_gt_medium.yaml
git add configs/splits/aist_only_smoke.yaml
git add configs/splits/aistpp_train_val.yaml
git add configs/splits/h36m_true_gt_aist_mixed_smoke.yaml
git add configs/splits/mix_h36m_aist_shelf.yaml

# 3. Stage scripts and experiments
git add experiments/diagnose_aistpp_smoke.py
git add experiments/run_aistpp_dlt_baseline.py
git add experiments/train_iskakov_baseline_shelf_campus.py
git add scripts/eval_variable_views_h36m_true_gt/
git add scripts/fetch_mpi_real_2d.sh
git add scripts/gpu_queue.sh
git add scripts/run_iskakov_aist_only_smoke_local_4090.sh
git add scripts/run_v25_aist_only_smoke_local_4090.sh
git add scripts/run_v57_h36m_true_gt_medium.sh
git add scripts/run_v80_aist_only_smoke_local_4090.sh
git add scripts/run_v80_h36m_true_gt_medium.sh
git add scripts/run_v80_h36m_true_gt_reg_v4_a800.sh
git add scripts/sota_baselines/

# 4. Review staged changes
git status --short
git diff --cached --stat

# 5. Commit with the message above
git commit -F- <<'EOF'
docs+infra: sync CVPR 2027 status, H36M true-GT results, and AIST++ smoke scripts

- Update AGENTS.md / CLAUDE.md with H36M true-GT availability, in-flight
  agents (agent-51, agent-67), and read-only A800-D rules.
- Refresh handoff and paper draft docs to remove unverifiable citations.
- Add H36M true-GT and AIST++ results/docs: results_true_gt_h36m.md,
  results_true_gt_shelf_campus.md, aistpp_smoke_diagnosis.md,
  results_aistpp_dlt_baseline.md, cvpr2027_status.md, v25_training_health.md,
  data_audit_summary_2026-08-11.md, roadmap_qwen3.8max.md.
- Add AIST++/H36M true-GT smoke configs and split files.
- Add evaluation / training scripts for v25/v57/v80 on H36M true-GT and
  AIST++, plus Iskakov baseline support and SOTA baseline helpers.
- Update A800 data inventory and Docker status docs.
- Record repository cleanup execution logs.

Does NOT include:
- models/mediapipe/ (large binary task files)
- tmp_mpi_debug_f0v0.png (transient debug image)
- archive_cleanup_20260811/ and cleanup_safe_20260811.sh (local cleanup artifacts)
- docs/git_status_summary.md (transient status snapshot)
EOF
```

## Excluded (Do Not Add)

| Path | Reason |
|------|--------|
| `models/mediapipe/` | Large binary task files; should be downloaded separately or handled via Git LFS if needed. |
| `tmp_mpi_debug_f0v0.png` | Transient debug visualization. |
| `archive_cleanup_20260811/` | Local archive of cleaned-up outputs; do not commit. |
| `cleanup_safe_20260811.sh` | One-off cleanup helper; not part of source tree. |
| `docs/git_status_summary.md` | Transient snapshot; do not commit. |

## Verification Checklist

- [ ] `git status --short` shows only the files listed under "Files Included".
- [ ] No `models/mediapipe/` files, `tmp_mpi_debug_f0v0.png`, or cleanup archive files are staged.
- [ ] `git diff --cached --stat` looks reasonable (mostly docs/configs/scripts).
- [ ] Commit message body fits within 72 columns per line.
- [ ] Local RTX 4090 GPU is not needed for this commit operation (no GPU task launched).

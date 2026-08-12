# Commit Bundle — 2026-08-11 v2

> Generated: 2026-08-11T13:14:00+08:00
> Branch: `main`
> HEAD: `3f81edf` — *Extend view-dropout ablation note with Shelf/Campus confirmation*

## Bundle Purpose

This is the **final** staging bundle for the H36M true-GT / AIST++ integration work and CVPR 2027 documentation update. It supersedes `docs/commit_bundle_2026-08-11.md` and captures every new/modified doc, config, script, and experiment helper that should enter the repository, while continuing to exclude transient artifacts and large binaries.

## Suggested Commit Message

```text
docs+infra: final H36M true-GT / AIST++ docs, configs, and scripts for CVPR 2027

- Update AGENTS.md / CLAUDE.md with H36M true-GT status, read-only A800-D
  rules, and current GPU occupancy (v57 H36M true-GT medium running).
- Refresh paper draft, handoff, and roadmap docs to reflect corrected
  non-circular protocol and AIST++ integration progress.
- Add H36M true-GT result docs: results_true_gt_h36m.md,
  results_true_gt_shelf_campus.md.
- Add AIST++ result and diagnosis docs: aistpp_smoke_diagnosis.md,
  results_aistpp_dlt_baseline.md, results_aistpp_iskakov_full.md,
  results_mpi_detected_dlt.md, v25_divergence_diagnosis.md,
  v25_training_health.md.
- Add status/audit docs: cvpr2027_status.md,
  data_audit_summary_2026-08-11.md, cleanup_plan.md,
  cleanup_executed.md, a800_docker_status.md, a800_data_inventory.md,
  github_branch_cleanup.md, github_branch_cleanup_executed.md,
  roadmap_qwen3.8max.md.
- Add H36M true-GT and AIST++ smoke configs and split files, including
  mix_true_gt_v2 and aistpp_train_val_mixed splits variants.
- Add evaluation / training / baseline scripts for v25/v57/v80 on H36M
  true-GT and AIST++, Iskakov full-run script, mix-true-gt generator,
  and SOTA baseline helpers.
- Record repository cleanup execution logs and GitHub branch cleanup.

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
- `docs/results_aistpp_iskakov_full.md`
- `docs/results_mpi_detected_dlt.md`
- `docs/results_true_gt_h36m.md`
- `docs/results_true_gt_shelf_campus.md`
- `docs/roadmap_cvpr2027.md`
- `docs/roadmap_qwen3.8max.md`
- `docs/v25_divergence_diagnosis.md`
- `docs/v25_training_health.md`

### New configs

- `configs/benchmark_v57_h36m_true_gt_medium.yaml`
- `configs/splits/aist_only_smoke.yaml`
- `configs/splits/aistpp_train_val.yaml`
- `configs/splits/aistpp_train_val_mixed.yaml`
- `configs/splits/h36m_true_gt_aist_mixed_smoke.yaml`
- `configs/splits/mix_h36m_aist_shelf.yaml`
- `configs/splits/mix_true_gt_v2.yaml`

### Experiments

- `experiments/diagnose_aistpp_smoke.py`
- `experiments/eval_variable_views.py`
- `experiments/run_aistpp_dlt_baseline.py`
- `experiments/train_iskakov_aistpp_full.py`
- `experiments/train_iskakov_baseline_shelf_campus.py`

### Scripts

- `scripts/eval_variable_views_h36m_true_gt/`
- `scripts/fetch_mpi_real_2d.sh`
- `scripts/generate_mix_true_gt_v2.py`
- `scripts/gpu_queue.sh`
- `scripts/run_iskakov_aist_only_smoke_local_4090.sh`
- `scripts/run_iskakov_aistpp_full_local_4090.sh`
- `scripts/run_mpi_dlt_baseline.py`
- `scripts/run_v25_aist_only_smoke_local_4090.sh`
- `scripts/run_v25_aistpp_train_val_local_4090.sh`
- `scripts/run_v25_mix_true_gt_v2_medium_local_4090.sh`
- `scripts/run_v25_v80_aistpp_train_val_local_4090.sh`
- `scripts/run_v57_h36m_true_gt_medium.sh`
- `scripts/run_v80_aist_only_smoke_local_4090.sh`
- `scripts/run_v80_aistpp_train_val_local_4090.sh`
- `scripts/run_v80_h36m_true_gt_medium.sh`
- `scripts/run_v80_h36m_true_gt_reg_v4_a800.sh`
- `scripts/sota_baselines/`
- `scripts/wait_idle_then_v57.sh`

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
git add docs/results_aistpp_dlt_baseline.md docs/results_aistpp_iskakov_full.md
git add docs/results_mpi_detected_dlt.md
git add docs/results_true_gt_h36m.md docs/results_true_gt_shelf_campus.md
git add docs/roadmap_cvpr2027.md docs/roadmap_qwen3.8max.md
git add docs/v25_divergence_diagnosis.md docs/v25_training_health.md

# 2. Stage new configs
git add configs/benchmark_v57_h36m_true_gt_medium.yaml
git add configs/splits/aist_only_smoke.yaml
git add configs/splits/aistpp_train_val.yaml
git add configs/splits/aistpp_train_val_mixed.yaml
git add configs/splits/h36m_true_gt_aist_mixed_smoke.yaml
git add configs/splits/mix_h36m_aist_shelf.yaml
git add configs/splits/mix_true_gt_v2.yaml

# 3. Stage experiments
git add experiments/diagnose_aistpp_smoke.py
git add experiments/eval_variable_views.py
git add experiments/run_aistpp_dlt_baseline.py
git add experiments/train_iskakov_aistpp_full.py
git add experiments/train_iskakov_baseline_shelf_campus.py

# 4. Stage scripts
git add scripts/eval_variable_views_h36m_true_gt/
git add scripts/fetch_mpi_real_2d.sh
git add scripts/generate_mix_true_gt_v2.py
git add scripts/gpu_queue.sh
git add scripts/run_iskakov_aist_only_smoke_local_4090.sh
git add scripts/run_iskakov_aistpp_full_local_4090.sh
git add scripts/run_mpi_dlt_baseline.py
git add scripts/run_v25_aist_only_smoke_local_4090.sh
git add scripts/run_v25_aistpp_train_val_local_4090.sh
git add scripts/run_v25_mix_true_gt_v2_medium_local_4090.sh
git add scripts/run_v25_v80_aistpp_train_val_local_4090.sh
git add scripts/run_v57_h36m_true_gt_medium.sh
git add scripts/run_v80_aist_only_smoke_local_4090.sh
git add scripts/run_v80_aistpp_train_val_local_4090.sh
git add scripts/run_v80_h36m_true_gt_medium.sh
git add scripts/run_v80_h36m_true_gt_reg_v4_a800.sh
git add scripts/sota_baselines/
git add scripts/wait_idle_then_v57.sh

# 5. Review staged changes
git status --short
git diff --cached --stat

# 6. Commit with the message above
git commit -F- <<'EOF'
docs+infra: final H36M true-GT / AIST++ docs, configs, and scripts for CVPR 2027

- Update AGENTS.md / CLAUDE.md with H36M true-GT status, read-only A800-D
  rules, and current GPU occupancy (v57 H36M true-GT medium running).
- Refresh paper draft, handoff, and roadmap docs to reflect corrected
  non-circular protocol and AIST++ integration progress.
- Add H36M true-GT result docs: results_true_gt_h36m.md,
  results_true_gt_shelf_campus.md.
- Add AIST++ result and diagnosis docs: aistpp_smoke_diagnosis.md,
  results_aistpp_dlt_baseline.md, results_aistpp_iskakov_full.md,
  results_mpi_detected_dlt.md, v25_divergence_diagnosis.md,
  v25_training_health.md.
- Add status/audit docs: cvpr2027_status.md,
  data_audit_summary_2026-08-11.md, cleanup_plan.md,
  cleanup_executed.md, a800_docker_status.md, a800_data_inventory.md,
  github_branch_cleanup.md, github_branch_cleanup_executed.md,
  roadmap_qwen3.8max.md.
- Add H36M true-GT and AIST++ smoke configs and split files, including
  mix_true_gt_v2 and aistpp_train_val_mixed split variants.
- Add evaluation / training / baseline scripts for v25/v57/v80 on H36M
  true-GT and AIST++, Iskakov full-run script, mix-true-gt generator,
  and SOTA baseline helpers.
- Record repository cleanup execution logs and GitHub branch cleanup.

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
- [ ] No GPU task was launched by this commit operation.

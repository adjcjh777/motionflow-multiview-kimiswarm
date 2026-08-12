# Circular-Label H36M Reference Audit

> **Date:** 2026-08-11  
> **Scope:** `docs/` and `scripts/` in `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`  
> **Auditor:** agent-207  
> **Canonical true-GT source:** `docs/results_true_gt_h36m.md`, `docs/results_iskakov_h36m_true_gt.md`, `docs/results_v80_h36m_true_gt.md`

## Background

The old H36M `.npz` files (`data/h36m_hf/*.npz`, `data/webbridge/h36m*.npz`) contain **circular labels**: their `joints_3d` are the unweighted DLT triangulation of the input 2D keypoints, so `direct MJE  0 mm`. Any H36M MPJPE result much below the DLT baseline on true mocap (≈ 25–30 mm) is therefore an artifact of the circular protocol.

True-GT H36M is now in:

- `data/h36m_true_gt/*_multiview_m.npz`
- manifest `configs/splits/h36m_true_gt_standard.yaml`
- standard protocol: train S1,S5,S6,S7,S8 → test S9,S11

This audit lists **remaining references in `docs/` and `scripts/`** to the old circular-label H36M numbers or data paths, and suggests how to fix or deprecate them.

---

## 1. Docs that still present old circular H36M numbers as if valid

| File | Line(s) | Stale claim | Suggested fix |
|------|---------|-------------|---------------|
| `docs/icra_cvpr_2027_paper_story.md` | 190 | "On … Human3.6M it reaches **0.62 mm** MPJPE." | Replace with true-GT H36M numbers (Iskakov 23.35 mm, v80 39.98 mm, v25 72.80 mm) or add the same superseded warning as the other paper outlines. |
| `docs/literature_novelty_positioning.md` | 19 | H36M S5/Act2: **0.62 mm** and **0.70 mm** for CamPE + GraphJR. | Annotate as circular-label era; reference `docs/results_true_gt_h36m.md` for current numbers. |
| `docs/paper_outline_v25_icra_cvpr_2027.md` | 33, 139 | Abstract and results table claim H36M **0.62 mm** / **5.24 mm** / **6.20 mm**. | File already has a superseded warning at the top, but the body still quotes the old numbers. Replace the claims with true-GT numbers or a "circular-label artifact" note. |
| `docs/results_h36m_v1.md` | entire file | Reports H36M val MPJPE of **1.48–2.12 mm** and DLT matching within ~2 mm. | Add a superseded header explaining these numbers are on circular labels; do not use for model selection. |
| `docs/results_h36m_v2.md` | entire file | Reports H36M val MPJPE **3.10–3.65 mm** and DLT matching within ~3 mm. | Add a superseded header; archive or redirect to true-GT results. |
| `docs/results_h36m_v1_metric.md` | entire file | Reports H36M errors in **meters** as low as **0.0004 m** (0.4 mm). | Add superseded header; these are circular-label numbers. |
| `docs/results_h36m_v2_dense_graph_a800.md` | 6–13 | Reports H36M val/test **15.03–24.04 mm** on a model trained on WebBridge H36M. | Add superseded header and note the labels were circular. |
| `docs/results_icra_cvpr_2027.md` | 17–21 | H36M table lists **5.24 mm** / **6.20 mm** MPJPE for cross-view residual models. | Add a circular-label caveat or replace with true-GT leaderboard. |

### Notes on correctly contextualized references

The following docs mention the old **0.62 mm** number only to contrast it with the true-GT protocol. They are intentionally warning readers and do **not** need changes:

- `docs/results_true_gt_h36m.md:175`
- `docs/leaderboard_summary_2026-08-11.md:101`
- `docs/roadmap_cvpr2027.md:48`
- `docs/paper_draft_icra_cvpr_2027.md:6,180` (explicit data-foundation caveat)

---

## 2. Scripts still pointing to old circular H36M data paths

These scripts hard-code `data/webbridge/h36m_meters/`, `data/webbridge/h36m/`, or `data/h36m_hf/`. Running them for model selection, training, or evaluation will reproduce circular-label numbers.

### 2.1 High severity — training / eval / benchmarking on circular H36M

| File | Line(s) | Problem | Suggested fix |
|------|---------|---------|-------------|
| `scripts/auto_eval_when_ready.sh` | 114 | `VAL_H36M="data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz"` | Point at `data/h36m_true_gt/s_09_acts_02_..._multiview_m.npz` and the standard manifest. |
| `scripts/benchmark_v26_full_queue_local_4090.sh` | 5 | `H36M="data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz"` | Use `data/h36m_true_gt/` and `configs/splits/h36m_true_gt_standard.yaml`. |
| `scripts/eval_crossview_pp_h36m_wsl.sh` | 12 | `--dataset data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz` | Update to true-GT val file. |
| `scripts/eval_mixed_pp_w05_wsl.sh` | 22 | `--dataset data/webbridge/h36m_meters/s_01_acts_07_multiview_m.npz` | Update to true-GT files. |
| `scripts/eval_v25_h36m_wsl.sh` | 15 | `DATASET=${DATASET:-data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz}` | Default to `data/h36m_true_gt/...`. |
| `scripts/run_crossview_pp_h36m_full_ppw005_wsl.sh` | 12, 17 | Train/val on `data/webbridge/h36m_meters/...` | Migrate to true-GT train/val manifest. |
| `scripts/run_crossview_pp_h36m_full_wsl.sh` | 12, 17 | Same as above. | Same as above. |
| `scripts/run_crossview_pp_h36m_wsl.sh` | 13, 18 | Same as above. | Same as above. |
| `scripts/run_full_v5_benchmark.py` | 11 | `--h36m data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz` | Use true-GT path/manifest. |
| `scripts/run_h36m_crossview_residual.sh` | 11 | `DATA_DIR="data/webbridge/h36m"` | Use `data/h36m_true_gt/`. |
| `scripts/run_mixed_focal_wsl.sh` | 17–21 | Lists `data/webbridge/h36m_meters/s_01_acts_02..06_multiview_m.npz` for mixed training. | Replace with true-GT train files or deprecate. |
| `scripts/run_mixed_mpi_h36m_focal_wsl.sh` | 17–21 | Same as above. | Same as above. |
| `scripts/run_mixed_pp_wsl.sh` | 17–21 | Same as above. | Same as above. |
| `scripts/run_ssl_pretrain_h36m_wsl.sh` | 8–14 | Train/val on `data/webbridge/h36m_meters/...` | Use true-GT files or mark as deprecated. |
| `scripts/run_ssl_pretrain_h36m_full_wsl.sh` | 8 | `TRAIN_DIR="data/webbridge/h36m"` | Use `data/h36m_true_gt/`. |
| `scripts/run_webbridge_mixed_smoke_wsl.sh` | 13, 18 | Mixed smoke uses `data/webbridge/h36m_meters/...` | Switch to true-GT / non-circular mix manifest. |
| `scripts/convert_h36m_to_meters.sh` | 3–16 | Converts `data/webbridge/h36m/` and `data/webbridge/h36m_corrected/` into the meter-scale circular set. | **Obsolete** — true-GT files are already in meters. Archive or delete. |

### 2.2 Medium severity — visualization / failure-analysis / examples on circular H36M

| File | Line(s) | Problem | Suggested fix |
|------|---------|---------|-------------|
| `scripts/analyze_v25_failures.py` | 19 | Default `--dataset data/webbridge/h36m_meters/s_11_acts_02_multiview_m.npz`. | Update default to a true-GT file or make the argument required. |
| `scripts/visualize_model_comparison.py` | 29 | `--dataset data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz` | Update to true-GT example. |
| `scripts/visualize_multiview_pose.py` | 15 | `--sample data/webbridge/h36m_meters/s_01_acts_10_multiview_m.npz` | Update to true-GT example. |
| `scripts/visualize_multiview_triangulation.py` | 18 | `--sample data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz` | Update to true-GT example. |
| `scripts/visualize_v25_geometry_attention.py` | 23 | `--sample data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz` | Update to true-GT example. |
| `scripts/visualize_variable_view_failure.py` | 28 | `--dataset data/webbridge/h36m/S9/acts_02_multiview_m.npz` | Update to true-GT file. |

### 2.3 Low severity / intentional diagnostic references

| File | Line(s) | Problem | Suggested fix |
|------|---------|---------|-------------|
| `scripts/check_true_gt_reprojection.py` | 15 | Docstring example uses `data/h36m_hf/s_01_act_02_multiview.npz`. | Update example to `data/h36m_true_gt/s_01_act_02_multiview_m.npz`. |
| `scripts/diagnose_circular_labels.py` | 4, 11, 12 | Uses `data/h36m_hf/*.npz` as the canonical example of circular labels. | **Intentional / keep.** This is the diagnostic tool; the example path is appropriate. |
| `scripts/fetch_h36m_true_gt.py` | 435–437 | Suggested next step writes output to `data/h36m_hf/s_01_act_02_multiview.npz`. | Change suggested output path to `data/h36m_true_gt/...`. |

---

## 3. Docs that still mention old circular H36M data paths

The `docs/swarm_iter*/` and `docs/swarm_iter_next/` directories contain many historical references to `data/h36m_hf/`, `data/webbridge/h36m/`, and `data/webbridge/h36m_meters/`. These are generally **archive notes** from before the true-GT pivot, but they can mislead a new collaborator. Representative examples:

| File | Why it matters |
|------|----------------|
| `docs/data_foundation_blocker.md` | Correctly explains the circular-label problem; keep, but ensure it points to the true-GT protocol doc. |
| `docs/mixed_loader_audit_v25.md:59–60` | Mentions `data/webbridge/h36m_meters/...` range issues. Should redirect to `data/h36m_true_gt/`. |
| `docs/proposals/benchmark_protocol.md:97` | JSON example lists `data/webbridge/h36m_meters/s_11_acts_02_multiview_m.npz`. Update to true-GT path. |
| `docs/swarm_iter11_h36m_experiment_pipeline_report.md` | Describes the old `data/webbridge/h36m/` pipeline. Add a superseded header. |
| `docs/swarm_iter23/webbridge_data_report.md` | Inventory of `data/webbridge/h36m_meters/...` files. Add a note that these are circular and superseded. |
| `docs/training_protocol_h36m_true_gt.md` | **Correct** — explicitly forbids old paths and points to `data/h36m_true_gt/`. Use as the canonical protocol doc. |

### Quick way to find more

```bash
# All remaining references to old circular H36M paths in docs/
rg -n "data/webbridge/h36m|data/h36m_hf/" docs/ > docs/circular_label_paths_in_docs.txt

# All remaining references in scripts/
rg -n "data/webbridge/h36m|data/h36m_hf/" scripts/ > docs/circular_label_paths_in_scripts.txt
```

---

## 4. Deprecated result docs to archive or annotate

These docs record results that were measured on circular-label H36M. They should be kept for history but clearly marked as **superseded / circular-label era**:

- `docs/results_h36m_v1.md`
- `docs/results_h36m_v2.md`
- `docs/results_h36m_v1_metric.md`
- `docs/results_h36m_v2_dense_graph_a800.md`
- `docs/results_icra_cvpr_2027.md` (H36M section)
- `docs/paper_outline_v25_icra_cvpr_2027.md`
- `docs/icra_cvpr_2027_paper_story.md`
- `docs/literature_novelty_positioning.md`

Suggested header to add at the top of each:

```markdown
> ⚠️ **SUPERSEDED (2026-08-10).** The H36M numbers in this document were measured on
> circular-label `.npz` files (`data/h36m_hf/`, `data/webbridge/h36m*.npz`).
> They are not comparable to true mocap ground-truth results.
> Current true-GT numbers are in `docs/results_true_gt_h36m.md`.
```

---

## 5. Recommended repository actions

1. **Fix high-severity scripts** (Section 2.1): migrate them to `data/h36m_true_gt/` and `configs/splits/h36m_true_gt_standard.yaml`, or delete them if they are no longer useful.
2. **Add superseded headers** to all docs in Section 4.
3. **Update default/example paths** in `scripts/analyze_v25_failures.py`, `scripts/visualize_*.py`, `scripts/check_true_gt_reprojection.py`, and `scripts/fetch_h36m_true_gt.py`.
4. **Archive or delete `scripts/convert_h36m_to_meters.sh`** — it only processes circular-label files.
5. **Add a CI/pre-commit guard** that rejects new references to `data/webbridge/h36m_meters/` or `data/h36m_hf/` in scripts, unless the file is `scripts/diagnose_circular_labels.py` or another explicitly allowed diagnostic.
6. **Keep `scripts/diagnose_circular_labels.py`** unchanged — it is the canonical diagnostic and is allowed to reference `data/h36m_hf/`.

---

## 6. Blockers / caveats

- This audit is **read-only**; no scripts or docs were modified. Applying the suggested fixes requires deciding which old scripts are worth migrating versus deleting.
- Some old scripts (e.g. `run_crossview_pp_h36m_*.sh`) may be superseded by newer true-GT scripts such as `scripts/run_v25_h36m_true_gt_medium_local_4090.sh`. Verify before investing effort in migration.
- The swarm-iter historical docs are intentionally left as archive material; only the top-level / paper-facing docs need urgent annotation.

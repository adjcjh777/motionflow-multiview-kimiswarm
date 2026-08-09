# Agent-16: A800 Historical Baseline Summary for v47 Temporal Aggregation

**Date:** 2026-08-09  
**Scope:** Read-only review of completed and running A800-D experiments to establish baselines for v47 temporal aggregation.  
**Sources inspected:**

- `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/` — training logs and `leaderboard.json`
- `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/docs/a800_results_summary.md`
- `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/docs/results_snapshot_2026_08_09.md`
- `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/docs/a800_queue_status.md`
- `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/docs/results_variable_views_curriculum.md`
- `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/docs/results_variable_views_quick.md`

> **Note on path:** The requested directory `/mnt/nvme0n1p1/zhangzy/projects` does not contain the MotionFlow experiment outputs. The active A800-D repository with the historical results is at `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`.

---

## 1. Best completed A800 runs (sorted by best val_MPJPE)

| Run | Best val_MPJPE (mm) | Epoch-1 val_MPJPE (mm) | Status | Notes |
|---|---:|---:|---|---|
| v25 geometry fusion full | **17.17** | 17.17 | Completed | Best A800 baseline; overfits after epoch 1 (epoch 2 = 59.14 mm) |
| v25 geometry fusion small | 18.31 | 18.31 | Completed | Overfits after epoch 1 |
| v31 domain balanced | 25.90 | 25.90 | Completed | Early stopping at epoch 6 |
| v34 HMSP + VJGN stack | 24.52 | 24.52 | Completed | Strongest v31-v34 stack on A800 |
| v34 view-joint graph network | 25.87 | 25.87 | Completed | — |
| v31 hierarchical more dropout | 26.97 | 26.97 | Completed | — |
| v32 combined | 26.49 | 26.49 | Completed | — |
| v32 trajectory consistency | 26.51 | 26.51 | Completed | — |
| v32 ray attention | 26.58 | 26.58 | Completed | — |
| v33 ray conditioned attention | 26.85 | 26.85 | Completed | — |
| v34 geometry VJGN | 27.59 | 27.59 | Completed | — |
| v34 geometry VJGN n_layers=1 | 28.51 | 28.51 | Completed | — |
| v34 geometry VJGN dropout=0.1 | 30.91 | 30.91 | Completed | — |
| v34 HMSP + geometry VJGN stack | 27.89 | 27.89 | Completed | — |
| v33 HMSP | 27.54 | 27.54 | Completed | — |
| v33 uncertainty-aware triangulation | 27.57 | 27.57 | Completed | — |
| v32 physical alignment | 27.75 | 27.75 | Completed | — |
| v31 physical floor only | 28.41 | 28.41 | Completed | — |
| v33 outlier view rejection | 30.57 | 30.57 | Completed | — |
| v31 geometry attention | 33.69 | 33.69 | Completed | — |
| v31 outlier view adaptive | 37.87 | 37.87 | Completed | — |
| v33 combined all three | 37.14 | 37.14 | Completed | — |
| v33 combined all three fixed | 38.13 | 38.13 | Completed | — |
| v33 combined all three + HMSP | 52.96 | 52.96 | Completed | — |

---

## 2. Priority comparison runs (v44 decision queue)

| Run | Status | Epoch-1 / Best val_MPJPE | Notes |
|---|---|---|---|
| v25 geometry fusion all-train baseline | Running (step ~350) | N/A | Training on full WebBridge/H36M/MPI mixed manifest |
| v25 + physical + domain | Crashed at startup | N/A | `ValueError: Unknown dataset '3dpw'` — manifest/loader mismatch |
| v42 v36 + physical + domain (no v37) | Crashed at startup | N/A | `FileNotFoundError` for Windows-style path `data\webbridge\...` |
| v43 adaptive per-node residual (base) | Crashed at startup | N/A | Same `FileNotFoundError` for WebBridge data path |
| v43 adaptive per-node residual (scaled) | Crashed at startup | N/A | Same `FileNotFoundError` |

**Implication:** The A800 priority runs needed for the v44 architecture decision are currently blocked by data-path / manifest issues. Until those are resolved, **v25 geometry fusion full at 17.17 mm remains the only reliable A800 baseline** for v47 to beat or regress against.

---

## 3. Sparse-view / variable-view historical baselines

These numbers come from the legacy variable-view evaluation on MPI-INF-3DHP and are the closest A800-recorded sparse-view baselines available before v46-SVG has produced results.

### Curriculum average (5 subsets)

| k views | MPJPE@k (mm) | std (mm) |
|---:|---:|---:|
| 2 | 79.55 | 34.83 |
| 3 | 39.98 | 9.07 |
| 4 | 35.23 | 12.80 |
| 6 | 24.11 | 5.48 |
| 10 | 17.06 | 1.19 |
| 14 (full) | 9.47 | 0.00 |

### Quick single-subset

| k views | MPJPE@k (mm) |
|---:|---:|
| 2 | 98.16 |
| 3 | 75.02 |
| 4 | 68.71 |
| 7 | 38.88 |
| 14 (full) | 14.72 |

**Takeaway:** With 2 views, MPJPE is ~80–98 mm; with 3 views it drops to ~40–75 mm. v47 temporal aggregation is expected to close much of this gap relative to full-view performance.

---

## 4. Queue status relevant to v47

- **v45-AGF A800 run:** `v45_adaptive_geometry_fusion_all_train` is queued behind the priority comparison runs.
- **v46-SVG A800 run:** `v46_sparse_view_generalization_on_v45` is queued; no A800 results yet. Local RTX 4090 smoke is pending.
- **v47-temporal A800 run:** `v47_temporal_aggregation_on_v46` is already registered in `scripts/launch_v33_a800_queue.py` but is blocked until v45/v46 complete.

---

## 5. Implications for v47 design / evaluation

1. **Primary baseline:** The v47 full-view target must not regress the v25 geometry-fusion full baseline of **17.17 mm** on full views.
2. **Sparse-view target:** v47 should improve on the historical 2-view / 3-view numbers above; a realistic interim goal is **MPJPE@2 < 60 mm** and **MPJPE@3 < 35 mm** while keeping full-view performance within ~1–2 mm of v25.
3. **Staging risk:** Because v46 has not yet produced A800 numbers, the v47 A800 comparison will initially be against v25 and the historical variable-view baselines rather than a v46 A800 checkpoint.
4. **Data path blockers:** The crashes in v25+physical+domain, v42, and v43 suggest that any v47 A800 run must use a manifest that is validated on the A800 filesystem (avoiding Windows paths and unsupported dataset keys such as `3dpw`).

---

*Report generated by Agent-16 (ANALYZE) for the v47 temporal aggregation swarm.*

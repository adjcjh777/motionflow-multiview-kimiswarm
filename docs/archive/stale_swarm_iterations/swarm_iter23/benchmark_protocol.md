# H36M / MPI-INF-3DHP Validation Split Analysis & CVPR/ICRA 2027 Benchmark Protocol

**Scope:** Define a single, reproducible benchmark protocol for the CVPR/ICRA 2027 paper that uses the existing H36M and MPI-INF-3DHP train/val/test YAML splits.  The protocol is intended for model selection (val) and final reporting (test), and it reuses the existing `scripts/run_full_benchmark.py` driver.

---

## 1. Current split definitions

The project stores canonical splits under `configs/splits/`:

| Dataset | Split YAML | Train | Val | Test |
|---|---|---|---|---|
| Human3.6M | `configs/splits/webbridge_h36m_train_val.yaml` | S1, all 15 actions | S9, all 15 actions | S11, all 15 actions |
| MPI-INF-3DHP | `configs/splits/webbridge_mpiinf3dhp_train_val_test.yaml` | S1, S3–S8 (both seqs) | S2 Seq1 | TS1–TS6 (official test set) |

### 1.1 H36M split analysis

| Split | Sequences | Frames | Notes |
|---|---:|---:|---|
| Train (S1) | 15 | 62,094 | Single subject; all 15 actions (act02–act16) |
| Val (S9) | 15 | 83,759 | Same actions as train; different subject |
| Test (S11) | 15 | 57,971 | Same actions; different subject |
| **Total** | **45** | **203,824** | — |

**Findings**

- **No subject overlap** between train/val/test, so there is no identity leakage.
- **Action overlap is total**: every action appears in all three sets.  This is the standard H36M protocol, but it means generalisation is measured across subjects, not across actions.
- The val set is **larger than the train set** (~35% more frames).  This is acceptable for model selection, but early stopping / hyper-parameter tuning should not assume val is a small subset.
- Compared with the wider literature (which often trains on S1, S5, S6, S7, S8), training on **only S1** is a much smaller training corpus.  Any number reported on S11 with this split is therefore not directly comparable to standard H36M baselines unless the same S1-only training is used.

### 1.2 MPI-INF-3DHP split analysis

| Split | Sequences | Frames | Notes |
|---|---:|---:|---|
| Train | 14 | 119,010 | Subjects S1, S3–S8, two sequences each |
| Val (S2 Seq1) | 1 | 6,502 | Single sequence, same capture conditions as train |
| Test (TS1–TS6) | 6 | 24,888 | Official test set; local 3D GT is withheld |
| **Total** | **21** | **150,400** | — |

**Findings**

- **No subject overlap** with the official test set (TS1–TS6 are different subjects / environments).
- The val set is a **single sequence**, which makes it noisy for model selection (one sequence may not represent the full action distribution).  It should be treated as a *local 3D proxy* only.
- The official test set has **no usable local 3D ground truth** in the canonical `.npz` files (`joints_3d` is all zeros).  Final MPI-INF-3DHP numbers must be obtained from the official evaluation server or a separately released GT file.
- TS5 and TS6 are short (320 and 492 frames respectively); frame-weighted aggregation already handles this, but per-sequence tables should flag them.

---

## 2. Recommended CVPR/ICRA 2027 benchmark protocol

### 2.1 Principle

1. **Do not touch the test sets during training or model selection.**
2. **Use the YAML splits as the single source of truth** for train / val / test file lists.
3. **Report frame-weighted averages** per dataset and an overall aggregate across datasets.
4. **Run the official MPI-INF-3DHP test server** for the final MPI number; the local TS1–TS6 `.npz` files cannot provide MPJPE.

### 2.2 Benchmark manifest

Create `configs/benchmark_icra_cvpr_2027.yaml` (or use the existing `configs/benchmark_webbridge_*.yaml` patterns):

```yaml
model_config:
  model: crossview_residual_pp
  checkpoint: outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth
  source_n_views: 14
  clip_len: 13
  d: 64
  n_st_layers: 2
  residual_hidden: 128
  batch_size: 8
  gt_scale: 1.0
  camera_scale: 1.0
  val_stride: 1

datasets:
  - name: h36m_val
    path: configs/splits/webbridge_h36m_train_val.yaml
    split: val
  - name: h36m_test
    path: configs/splits/webbridge_h36m_train_val.yaml
    split: test
  - name: mpiinf3dhp_val
    path: configs/splits/webbridge_mpiinf3dhp_train_val_test.yaml
    split: val
  # MPI test is evaluated externally; include only as a placeholder / metadata entry.
  - name: mpiinf3dhp_test_official
    path: configs/splits/webbridge_mpiinf3dhp_train_val_test.yaml
    split: test
```

### 2.3 Running the benchmark

```bash
# H36M val + test, plus MPI val (local 3D)
python scripts/run_full_benchmark.py \
    --manifest configs/benchmark_icra_cvpr_2027.yaml \
    --out outputs/icra_cvpr_2027_full_benchmark

# Dry-run for CI / manifest validation
python scripts/run_full_benchmark.py \
    --manifest configs/benchmark_icra_cvpr_2027.yaml \
    --out outputs/dry_run \
    --dry-run
```

### 2.4 Metrics

Use `motionflow_mv/eval/metrics.py::compute_all_metrics` (units in mm):

- `mpjpe`
- `pa_mpjpe`
- `root_rel_mpjpe`
- `velocity_mpjpe`
- `pck@50mm`, `pck@100mm`, `pck@150mm`
- `pck_auc`
- `per_joint_mpjpe`
- `per_joint_pa_mpjpe`

Per-sequence metrics are aggregated with **frame counts as weights** to avoid short sequences dominating the average.

### 2.5 Reporting conventions

| Dataset | Primary metric | Reporting set | Special handling |
|---|---|---|---|
| H36M | MPJPE / PA-MPJPE | S11 test | Per-action breakdown recommended |
| MPI-INF-3DHP | PCK@150mm / AUC | Official test server | Local TS1–TS6 cannot be evaluated internally |
| Cross-dataset | MPJPE | Val (S9, S2 Seq1) | Use only for model selection / early stopping |

- **Main paper table:** report H36M test (S11) and MPI-INF-3DHP official test server results.
- **Ablation studies:** use H36M val (S9) and MPI val (S2 Seq1) only; never use the test sets.
- **Per-action H36M:** `scripts/run_full_benchmark.py` already stores per-sequence results; convert them to the standard 15-action table for the paper.

---

## 3. Risks & mitigations

| Risk | Mitigation |
|---|---|
| H36M val (S9) is larger than train (S1) | Treat it purely as an evaluation set; do not use it to tune capacity-sensitive hyper-parameters beyond early stopping. |
| MPI val is a single sequence | Use it only as a sanity check; final model selection for MPI should be guided by the official test server or a held-out internal split if created later. |
| MPI test set has no local 3D GT | Always submit to the official evaluation server for final numbers; the local `.npz` files are placeholders. |
| H36M train only uses S1 | Report this split explicitly in the paper; numbers are not comparable to full 5-subject H36M training without a separate baseline. |
| TS5/TS6 are very short | Use frame-weighted aggregation; show per-sequence tables to avoid hiding short-sequence effects. |

---

## 4. Next concrete step

Generate a complete dry-run benchmark report for the current v23 checkpoint on H36M val/test and MPI val using `scripts/run_full_benchmark.py --dry-run`; once the v23 run on A800-D produces a checkpoint, replace the dry-run with real inference and populate `docs/tables/icra2027/main_results.md`.

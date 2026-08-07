# Cross-Dataset Benchmark Protocol for ICRA/CVPR 2027

## 1. Goal

Define a single, reproducible protocol to evaluate any trained MotionFlow-MultiView checkpoint on the three canonical test sets used in the paper:

- **Human3.6M (H36M)** – `configs/splits/webbridge_h36m_train_val.yaml`
- **MPI-INF-3DHP** – `configs/splits/webbridge_mpiinf3dhp_train_val_test.yaml`
- **WebBridge (generalisation / mixed sequences)** – any YAML split or per-sequence `.npz` list

The protocol reports per-dataset and aggregated 3D pose metrics in a single JSON file.

## 2. Manifest format

A benchmark is described by a YAML manifest:

```yaml
model_config:
  model: crossview_residual_pp
  checkpoint: outputs/crossview_residual_d64_h128_full5.pth
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
  - name: h36m_test
    path: configs/splits/webbridge_h36m_train_val.yaml
    split: test
  - name: mpiinf3dhp_test
    path: configs/splits/webbridge_mpiinf3dhp_train_val_test.yaml
    split: test
  - name: webbridge_test
    path: configs/splits/webbridge_mpiinf3dhp_train_val_test.yaml
    split: test
```

The `path` may point to either:

1. A **YAML split file** (e.g. `configs/splits/*.yaml`). The script extracts the requested `split` list.
2. A single **canonical `.npz`** in the format used by `experiments/eval_full_metrics.py`.

## 3. Metrics

For every sequence the following metrics are computed by `motionflow_mv.eval.metrics.compute_all_metrics` (units are mm):

- `mpjpe`
- `pa_mpjpe`
- `root_rel_mpjpe`
- `velocity_mpjpe`
- `pck@50mm`, `pck@100mm`, `pck@150mm`
- `pck_auc`
- `per_joint_mpjpe`
- `per_joint_pa_mpjpe`
- `bone_length_error` (if skeleton parents are supplied)

Per-sequence results are stored, then per-dataset results are computed as frame-weighted averages.  Aggregating across *all* test sets gives a single leaderboard number.

## 4. Running the benchmark

```bash
python scripts/run_full_benchmark.py \
    --manifest configs/benchmark_icra_cvpr_2027.yaml \
    --out outputs/icra_cvpr_2027_full_benchmark
```

### Dry-run mode

For CI / test / planning, use `--dry-run`.  The script will:

- Parse the manifest.
- Determine every `.npz` to evaluate.
- Print (and record) the commands that would be executed.
- Write a JSON report with placeholder metrics but a valid schema.

```bash
python scripts/run_full_benchmark.py \
    --manifest configs/benchmark_icra_cvpr_2027.yaml \
    --out outputs/dry_run \
    --dry-run
```

## 5. Output schema

```json
{
  "manifest": "configs/benchmark_icra_cvpr_2027.yaml",
  "model_config": { ... },
  "datasets": [
    {
      "name": "h36m_test",
      "files": ["data/webbridge/h36m_meters/s_11_acts_02_multiview_m.npz", ...],
      "metrics": { "mpjpe": ..., "pa_mpjpe": ..., ... }
    }
  ],
  "summary": {
    "per_dataset": { "h36m_test": { ... }, ... },
    "overall": { "mpjpe": ..., "pa_mpjpe": ..., ... }
    }
}
```

All arrays are converted to plain lists for JSON serialization.

## 6. Implementation

The driver is `scripts/run_full_benchmark.py`.  It delegates per-sequence inference to the existing `experiments/eval_full_metrics.py` harness and re-uses `motionflow_mv.eval.metrics` for aggregation.  This avoids duplicating model-loading / clip-building logic.

## 7. Future work

- Add per-action / per-sequence breakdown for H36M.
- Add statistical testing (bootstrap confidence intervals) across seeds.
- Add MPI-INF-3DHP test-set protocol with TS1-TS6 official sequences.
- Add submission-format generation for the WebBridge public leaderboard.

# Sparse-View Robustness Evaluation Protocol

**Target:** CVPR / ICRA 2027 multi-view human pose estimation.  
**Scope:** Evaluate a fixed-view fusion model under variable active-view counts using `experiments/eval_variable_views.py` and the `MPJPE@k` protocol.  
**Last updated:** 2026-08-11

---

## 1. What this protocol measures

A fixed-view model is trained with a fixed number of cameras `V`.  At deployment it may only receive `k ≤ V` views.  The sparse-view robustness protocol reports:

- **MPJPE@k** – mean per-joint position error (mm) when only `k` views are active.
- **PA-MPJPE@k** – Procrustes-aligned MPJPE@k.
- **Root-Relative MPJPE@k** – pelvis-centred MPJPE@k.
- **Temporal jerk@k** – smoothness of the predicted sequence under view dropout.

The evaluation is inference-only, so it can run on CPU for smoke tests or on GPU for full benchmarks.  It does **not** require retraining or modifying the model.

---

## 2. Required inputs

### 2.1 Checkpoint and config

| Model family | Checkpoint | Config |
|---|---|---|
| Legacy ray-attention models (v25/v26/v46/v57/v80) | `outputs/<name>.pth` | optional `.config.json` |
| OmniMultiViewFusionV5 (v47/v48/...) | `outputs/<name>.pth` | `<name>.config.json` (required) |

The config is a flat JSON/YAML dictionary saved by the trainer.  It is used to rebuild the model architecture (number of layers, hidden size, view-count conditioning flags, etc.).

### 2.2 Dataset format

The evaluator reads a WebBridge-style `.npz` with the following keys:

```
points_2d   : (T, V, J, 2)  image-space keypoints in pixels
confidences : (T, V, J)    per-keypoint confidence
joints_3d   : (T, J, 3)     ground-truth 3D poses in metres
camera_K    : (V, 3, 3)     intrinsic matrices
camera_R    : (V, 3, 3)     rotation matrices
camera_t    : (V, 3)        camera translation vectors
```

`T` = frames, `V` = total views, `J` = joints.

### 2.3 Dataset manifest (optional but recommended)

For per-dataset reporting, create a manifest file with one line per dataset:

```text
h36m_s9_act11  data/h36m_true_gt/s_09_act_11_multiview_m.npz
h36m_s11_act2  data/h36m_true_gt/s_11_act_02_multiview_m.npz
mpi_s1_seq01   data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz
```

Lines starting with `#` and blank lines are ignored.

---

## 3. MPJPE@k metrics

For each requested view count `k`, the protocol:

1. Generates all `C(V, k)` view subsets when `V` is small, or samples `num_subsets_per_k` deterministic subsets otherwise.
2. Masks out inactive views by zeroing their confidence channels (see `motionflow_mv/fusion/variable_view_inference.py`).
3. Runs inference on non-overlapping clips of length `clip_len`.
4. Aggregates the following metrics per `k`:

| Metric | Symbol | Unit | Description |
|---|---|---|---|
| MPJPE@k | `mpjpe_at_k` / `mpjpe` | mm | Mean per-joint 3D error against GT |
| PA-MPJPE@k | `pa_mpjpe` | mm | Procrustes-aligned MPJPE |
| Root-Relative MPJPE@k | `root_rel_mpjpe` | mm | Root-centred MPJPE |
| Std. dev. | `std_mm` | mm | Standard deviation of per-subset MPJPE |
| Temporal jerk | `temporal_jerk` | mm | Mean magnitude of 3rd temporal derivative |
| Subsets evaluated | `n_subsets` | count | Number of view subsets used for this `k` |

The reported `MPJPE@k` is the **mean over all sampled subsets of size `k`**.  Per-subset results are also stored for debugging.

Reference implementations:

- `motionflow_mv/eval/mpjpe_at_k_protocol.py` – `evaluate_mpjpe_at_k`, `compute_mpjpe_at_k`
- `motionflow_mv/fusion/variable_view_inference.py` – `VariableViewInferenceWrapper`, `HardenedVariableViewInferenceWrapper`

---

## 4. Running `eval_variable_views.py`

### 4.1 CLI arguments

```text
--dataset              Path to a single .npz dataset (mutually exclusive with --dataset_manifest)
--dataset_name         Human-readable name for the dataset (default: file stem)
--dataset_manifest     Path to a manifest file with "<name> <path>" per line
--checkpoint           Model checkpoint (.pth)
--config               Saved training config (.json/.yaml) - required for omniview_v5
--model_class          One of: temporal_residual, crossview_residual,
                       crossview_residual_pp, crossview_residual_pp_visibility, omniview_v5
--n_views              Total view count for synthetic smoke test (default: 6)
--j                    Joint count for synthetic smoke test (default: 17)
--d                    Model feature dimension (default: 64)
--residual_hidden      Hidden dim for residual models (default: 128)
--clip_len           Temporal clip length (default: 9)
--n_temporal_layers  Temporal layers for legacy models (default: 2)
--min_views          Minimum active views (default: 2)
--max_views          Maximum active views (default: V)
--k_values           Explicit list, e.g. "2 3 4" (overrides min/max)
--num_subsets_per_k  Random subsets per k (default: 20; None = enumerate all)
--seed               RNG seed for subset sampling (default: 42)
--output_json        Optional JSON output path
--output_csv         Optional CSV output path
--v46_checkpoint / --v47_checkpoint  Side-by-side Omni v46 vs v47 comparison
--compare_v46_v47    Toggle v47 temporal aggregation on a single loaded model
```

### 4.2 CPU smoke test (no data / checkpoint required)

```bash
python experiments/eval_variable_views.py \
    --n_views 6 \
    --j 17 \
    --clip_len 9 \
    --num_subsets_per_k 10 \
    --seed 42 \
    --output_json tmp/variable_view_smoke/results.json \
    --output_csv  tmp/variable_view_smoke/results.csv
```

This uses a synthetic random dataset and exercises the variable-view inference logic on CPU.

### 4.3 Single-dataset benchmark

```bash
python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/omniview_fusion_v80_h36m_true_gt_medium.pth \
    --config outputs/omniview_fusion_v80_h36m_true_gt_medium.config.json \
    --dataset data/h36m_true_gt/s_11_act_02_multiview_m.npz \
    --dataset_name h36m_s11_act02 \
    --clip_len 13 \
    --min_views 2 \
    --max_views 4 \
    --num_subsets_per_k 50 \
    --seed 42 \
    --output_csv tmp/variable_view_single.csv \
    --output_json tmp/variable_view_single.json
```

### 4.4 Multi-dataset manifest benchmark

```bash
python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/omniview_fusion_v80_h36m_true_gt_medium.pth \
    --config outputs/omniview_fusion_v80_h36m_true_gt_medium.config.json \
    --dataset_manifest tmp/h36m_true_gt_val_manifest.txt \
    --clip_len 13 \
    --min_views 2 \
    --max_views 4 \
    --num_subsets_per_k 50 \
    --seed 42 \
    --output_csv tmp/variable_view_manifest.csv \
    --output_json tmp/variable_view_manifest.json
```

### 4.5 v46 vs v47 comparison

```bash
python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --v46_checkpoint outputs/v46_baseline.pth \
    --v47_checkpoint outputs/v47_temporal_svg.pth \
    --config outputs/v47_temporal_svg.config.json \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 \
    --k_values 2 3 4 \
    --output_csv tmp/v46_v47_comparison.csv \
    --output_json tmp/v46_v47_comparison.json
```

---

## 5. Variable-view inference strategies

The protocol supports two inference wrappers:

### 5.1 `VariableViewInferenceWrapper`

- Masks inactive views by zeroing confidence channels.
- Keeps the fixed `n_views` model input shape unchanged.
- Best for models with robust attention / weight heads that learn to ignore zero-confidence slots.

### 5.2 `HardenedVariableViewInferenceWrapper`

- Does everything the basic wrapper does, plus:
  1. Fills inactive camera slots with valid camera parameters (last-active or mean-active) so intrinsics/extrinsics are well-defined.
  2. Restricts graph-joint attention to active views when the model has a `graph_joint_attention` block.
  3. Falls back to confidence-weighted DLT when active views are below `min_views`.
- Automatically used by `eval_variable_views.py` when the model class name contains `OmniMultiViewFusionV5`.

Use the hardened wrapper for few-view (`k=2,3`) evaluation of graph-attention models; it mitigates the catastrophic failure mode observed when a 4-view model is fed fewer views.

---

## 6. Output formats

### 6.1 JSON (single dataset)

```json
{
  "mpjpe_at_k": {
    "2": 38.5042,
    "3": 29.1033,
    "4": 26.4211
  },
  "per_k": {
    "2": {
      "k": 2,
      "mpjpe": 38.5042,
      "std_mm": 4.1234,
      "n_subsets": 50,
      "temporal_jerk": 0.0123,
      "subsets": [[0, 1], [0, 2], ...]
    },
    "3": { ... },
    "4": { ... }
  }
}
```

### 6.2 CSV (single dataset)

```csv
k,mpjpe_at_k,mean_mm,std_mm,n_subsets,temporal_jerk
2,38.5042,38.5042,4.1234,50,0.0123
3,29.1033,29.1033,3.2100,50,0.0087
4,26.4211,26.4211,2.9876,50,0.0071
```

### 6.3 Per-dataset CSV (manifest mode)

```csv
dataset,k,mpjpe_at_k,mean_mm,std_mm,n_subsets,temporal_jerk
h36m_s9_act11,2,40.12,40.12,4.50,50,0.015
campus_act1,2,55.30,55.30,6.10,20,0.022
```

### 6.4 Comparison CSV (v46 vs v47)

```csv
dataset,k,v46_mpjpe_at_k,v47_mpjpe_at_k,delta_mm,delta_pct,v46_temporal_jerk,v47_temporal_jerk
h36m_s9_act11,2,40.12,38.50,1.62,4.04,0.015,0.012
```

### 6.5 Markdown reporting template

Use this table in paper or leaderboards:

```markdown
| Method | MPJPE@2 | MPJPE@3 | MPJPE@4 | Δ(k=2→4) |
|---|---|---|---|---|
| DLT baseline | 45.20 | 31.10 | 25.87 | 19.33 |
| v25 | 78.40 | 73.20 | 72.80 | 5.60 |
| v80 | 52.10 | 43.30 | 39.98 | 12.12 |
| v57 | TBD | TBD | TBD | — |
```

For per-dataset sparse-view results:

```markdown
| Dataset | k=2 | k=3 | k=4 | Subsets |
|---|---|---|---|---|
| H36M S9 | 40.12 | 30.45 | 27.10 | 50 |
| H36M S11 | 38.50 | 29.10 | 26.42 | 50 |
```

---

## 7. Recommended evaluation matrix

For a model trained on `V` views, evaluate the following `k` values:

| `V` | Recommended `k_values` |
|---|---|
| 4 (H36M, Campus) | `2 3 4` |
| 5 (Shelf) | `2 3 4 5` |
| 6 (some MPI-INF-3DHP setups) | `2 3 4 5 6` or `2 3 4 6` |

Use `num_subsets_per_k = 50` as a default.  Enumerate all subsets (`num_subsets_per_k` omitted) only when `C(V, k)` is small, e.g. for `V=4, k=2` there are only 6 subsets.

---

## 8. Safety and resource rules

- This protocol is **inference-only**.  It does not train.
- CPU smoke tests are safe to run at any time and do not touch data/checkpoints.
- Full dataset evaluation should be run on GPU only when no other training/evaluation job is active.  The local RTX 4090 can run **one GPU task at a time**.
- Do not write, start, or modify anything on A800-D `/mnt/nvme0n1/zhangzy/projects` or the A800 Docker `motionflow` service; those resources are read-only.

---

## 9. Acceptance criteria

1. `python experiments/eval_variable_views.py --n_views 6 --j 17 --clip_len 9` completes on CPU and prints MPJPE@k for `k=2..6`.
2. A real checkpoint + dataset run produces `output_json` with numeric `mpjpe_at_k` values for every requested `k`.
3. CSV output has the expected header and one row per `k`.
4. Re-running with the same `--seed` produces identical subset sampling and identical results.

---

## 10. References

- `experiments/eval_variable_views.py` – driver script
- `motionflow_mv/eval/mpjpe_at_k_protocol.py` – metric computation
- `motionflow_mv/fusion/variable_view_inference.py` – inference wrappers
- `motionflow_mv/eval/metrics.py` – underlying MPJPE / PA-MPJPE / root-relative metrics

---

## 11. Example: full H36M true-GT sparse-view benchmark

After training a model and creating a validation manifest, run:

```bash
CKPT="outputs/omniview_fusion_v80_h36m_true_gt_medium.pth"
CONFIG="outputs/omniview_fusion_v80_h36m_true_gt_medium.config.json"
MANIFEST="tmp/h36m_true_gt_val_manifest.txt"

python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --dataset_manifest "$MANIFEST" \
    --clip_len 13 \
    --min_views 2 --max_views 4 \
    --num_subsets_per_k 50 \
    --seed 42 \
    --output_csv "tmp/sparse_view_v80_h36m.csv" \
    --output_json "tmp/sparse_view_v80_h36m.json"
```

Then copy the generated CSV/JSON into the paper results directory and fill in the leaderboards in `docs/results_true_gt_h36m.md` and `docs/tables/icra2027/robustness.md`.

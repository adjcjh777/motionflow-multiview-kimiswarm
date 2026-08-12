# WebBridge Multi-Dataset Mixing: Train on MPI + H36M + Shelf + Campus Blend

## 1. Problem
The current mixed-data pipeline only registers MPI-INF-3DHP, AIST++, and Human3.6M, so Shelf/Campus cannot be folded into a single multi-dataset training run that would improve generalisation across different camera rigs and scene scales.

## 2. Hypothesis
Adding Shelf and Campus as first-class datasets in the existing mixed-dataset loader and per-dataset model heads, then training jointly with MPI-INF-3DHP and Human3.6M, will produce a single model that still matches MPI-INF-3DHP accuracy while improving robustness on small-view (3–5 camera) captures.

## 3. Method

### Data changes
- Extend `motionflow_mv/data/mixed_dataset.py`:
  - Add `shelf` and `campus` to `DATASET_REGISTRY` with `n_views=5 / 3`, `n_joints=17`.
  - Keep the existing padding to `(MAX_VIEWS=14, MAX_JOINTS=28)`; zero-confident padded slots already suppress contribution to triangulation.
- Verify that `motionflow_mv/data/webbridge_loader.py::convert_shelf_campus` already emits the canonical `.npz` layout (`points_2d`, `confidences`, `joints_3d`, `camera_K`, `camera_R`, `camera_t`).

### Model changes
- Extend `_DATASET_SPECS` in `motionflow_mv/fusion/ray_attention_temporal_mixed_residual_v1.py` to include:
  - `shelf`: 5 views, 17 joints
  - `campus`: 3 views, 17 joints
- The parent class in `motionflow_mv/fusion/ray_attention_temporal_model_mixed_v1.py` builds per-dataset `fusion_mlps` and `weight_heads` from this spec, so Shelf/Campus will automatically receive their own heads.
- `RayAttentionFusionModelTemporalMixedResidualPrincipalPoint` (`motionflow_mv/fusion/ray_attention_temporal_mixed_residual_principal_point_model.py`) inherits the same spec and will therefore support the four-way blend without new architecture code.

### Training script
- Create `experiments/train_webbridge_mixed_mpi_h36m_shelf_campus.py` by copying `experiments/train_mixed_dataset_principal_point.py` and adding:
  - `--shelf_train` and `--campus_train` arguments.
  - A `train_paths` dict with all four keys.
  - A `--smoke` flag that uses `d=32, residual_hidden=64, batch_size=4, train_samples=100, epochs=3`.

### Loss / training details
- Keep the same pose MSE + optional principal-point/focal correction losses used by `train_mixed_dataset_principal_point.py`.
- No new loss is introduced; dataset-specific heads already isolate skeleton/view differences.

## 4. Smoke-Test Plan
Run 3 epochs on a tiny subset using the new `--smoke` path:

```bash
python experiments/train_webbridge_mixed_mpi_h36m_shelf_campus.py \
  --mpi_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --h36m_train data/h36m_hf/s_01_acts_02_03_multiview_m.npz \
  --shelf_train data/shelf_campus/Shelf_Seq1/pseudogt_m.npz \
  --campus_train data/shelf_campus/Campus_Seq1/pseudogt_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --val_dataset mpi \
  --clip_len 13 \
  --smoke \
  --output outputs/webbridge_mixed_smoke.pth
```

Pass/fail criteria:
- Training completes 3 epochs without NaNs, OOM, or crashes.
- At least one sample from each of the four dataset ids is present in every training batch (batch-level mixing works).
- MPI-INF-3DHP validation MPJPE is finite and ≤ 30 mm on the small smoke split (we do not expect the 9.32 mm anchor number from 3 epochs on a tiny subset).
- Validation MPJPE is not more than 20 % worse than the same smoke run using MPI-only data.

## 5. Evaluation Plan
After the smoke passes, evaluate the checkpoint with:

- `experiments/eval_webbridge_mixed_mpi_h36m_shelf_campus.py` (new file, mirroring `eval_mixed_dataset_principal_point.py`) for:
  - MPI-INF-3DHP S2/Seq1 clean MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
  - Per-dataset validation loss for H36M, Shelf, and Campus if labelled validation `.npz`s are available.
- Compare against the MPI-only 9.32 mm anchor (`outputs/crossview_residual_d64_h128_full5.pth`) on the exact same split using `experiments/compare_sota_baselines.py`.
- Generate a short Markdown report at `docs/swarm_iter14/results_webbridge_mixed_smoke.md` with a table of metrics and a brief go/no-go recommendation.

## 6. Estimated GPU/CPU Cost on RTX 4090
- Smoke (3 epochs, 100 clips × 4 datasets, `d=32`, `batch_size=4`): ~5–10 minutes on a single RTX 4090, < 4 GB VRAM.
- Full run (20 epochs, 500 clips per dataset, `d=64`, `batch_size=8`): ~3–5 hours on a single RTX 4090, ~8–10 GB VRAM.
- Data conversion / pre-processing is CPU-only and already done for the canonical `.npz` files; total CPU cost < 1 minute.

## 7. Risks & Fallback
- **Skeleton ordering mismatch:** Shelf/Campus 17-joint ordering may not match Human3.6M. If smoke shows divergent per-joint errors, fall back to a small joint-remapping table in `mixed_dataset.py` before training.
- **Small-view overfitting:** Campus has only 3 views. If validation on MPI degrades, reduce the Shelf/Campus sampling weight to 0.25× and re-run.
- **PP/focal correction saturation:** The mixed trainer reuses the existing PP correction head; if it still saturates, disable PP correction for the smoke (`--pp_loss_weight 0`) and rely on ground-truth intrinsics.
- **Training instability from heterogeneous scale:** If loss diverges, add per-dataset batch-normalisation of 3D poses or scale Shelf/Campus units to meters (the pseudogt files are already in meters).

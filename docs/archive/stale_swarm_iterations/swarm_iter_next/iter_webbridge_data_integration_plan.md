# WebBridge Data Integration for Multi-Dataset Pose Fusion

**Direction:** `webbridge_data_integration`  
**Anchor to beat:** `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth` — **8.75 mm MPJPE** on MPI-INF-3DHP S2/Seq1 (PA-MPJPE 4.95 mm).  
**Current candidate in flight:** Bayesian triangulation (`experiments/train_bayesian_tri_pp_full_mpiinf3dhp.py`).  
**Date:** 2026-08-06  
**Author:** research swarm  

---

## 1. Motivation

The 8.75 mm anchor is a single-dataset MPI-INF-3DHP model. WebBridge already provides canonical `.npz` versions of Human3.6M, AIST++, Shelf/Campus, and MPI-INF-3DHP (`motionflow_mv/data/webbridge_loader.py`). This direction turns those heterogeneous captures into a single, calibrated, multi-dataset training signal.

Why this can beat 8.75 mm:

1. **Volume and diversity.** WebBridge adds >100k frames of calibrated multi-view data with different camera rigs (4-view H36M, 9-view AIST++, 14-view MPI, 3–5-view Shelf/Campus). A model trained on this mixture is less likely to overfit the MPI-INF-3DHP studio lighting and camera layout.
2. **Cross-dataset regularization.** H36M has a 17-joint subset that overlaps with MPI-INF-3DHP. Training on both forces the fusion backbone to learn skeleton-agnostic 3D geometry rather than dataset-specific artifacts.
3. **Robustness to view count.** Shelf/Campus has only 3–5 views. Including them explicitly trains the residual head to perform well under small-view regimes, where the 8.75 mm anchor has not been optimized.
4. **Calibrated domain gap is well-defined.** Because WebBridge stores `camera_K/R/t` in a common format, the camera-parameter branch of the PP model can learn a shared intrinsics/extrinsics prior across datasets.
5. **Failure-mode coverage.** The cross-view PP failure analysis (`docs/swarm_iter_next/failure_analysis_crossview_pp.md`) shows worst joints are hips, hands, and wrists. H36M provides richer upper-body motion; Shelf/Campus provides more natural walking sequences. Mixing should directly target these weak joints.

Estimated realistic gain: **5–15% MPJPE reduction on MPI-INF-3DHP** (target <8.3 mm) and a large improvement on cross-dataset transfer to H36M/Shelf/Campus.

---

## 2. Architecture

We propose a **WebBridge-mixed principal-point fusion model** that inherits from the anchor `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` and adds a cross-dataset mixing + consistency stage. The high-level flow is:

```
Canonical .npz (H36M/MPI/AIST/Shelf/Campus)
        |
        v
[skeleton map -> 17 common joints]
        |
        v
[mixed clip sampling per dataset]
        |
        v
RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
        |
        +-- Principal-point / focal correction
        +-- Per-frame ray features
        +-- Spatio-temporal (time × view × joint) attention
        +-- Weighted DLT triangulation
        +-- Residual MLP refinement
        |
        v
[Cross-dataset pose consistency loss on shared joints]
        |
        v
3D pose (17 joints, meters)
```

### 2.1 Skeleton unification

A new module `motionflow_mv/data/skeleton_maps.py` defines:

- `COMMON_17_JOINTS`: ordered list of 17 joint names (H36M convention).
- `MPIINF3DHP_TO_COMMON_17`: 28 -> 17 index map from MPI full skeleton to the common subset.
- `AIST_TO_COMMON_17`: identity / near-identity map.
- `SHELF_CAMPUS_TO_COMMON_17`: map from Shelf/Campus 17 joints to H36M order.

During dataset loading, `points_2d`, `confidences`, and `joints_3d` are re-indexed to 17 joints. The common subset focuses the model on joints that are present in every dataset, eliminating missing-keypoint padding issues.

### 2.2 Mixed-dataset loader

Extend `motionflow_mv/data/mixed_dataset.py` with a **WebBridge mode**:

- Register new entries in `DATASET_REGISTRY`:
  - `"h36m"`: 4 views, 17 joints
  - `"mpi"`: 14 views, 17 joints (after mapping)
  - `"aist"`: 9 views, 17 joints
  - `"shelf"`: 5 views, 17 joints
  - `"campus"`: 3 views, 17 joints
- `MAX_VIEWS` stays 14; small-view datasets are padded with zero-confidence slots exactly as today.
- Each sample returns `(x, y, K, R, t, dataset_id)`.

### 2.3 Model

Use the existing anchor as the backbone:

```python
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
```

Wrap or subclass it into `WebBridgeMixedCrossviewResidualPP`. The wrapper is minimal:

- Input: `(B, T, V, J, 3)` with `J=17`, `V=14` after padding.
- Forward: identical to the PP anchor; the only change is that batches contain heterogeneous camera rigs and zero-padded views.
- Output: `(pred_3d, weights, pp_delta)` where `pred_3d` is `(B, T, 17, 3)`.

The zero-confidence masking in the anchor (`weights = sigmoid(logits) * confidences`) already handles padded views correctly.

### 2.4 Cross-dataset consistency loss

Let `B_s` be the number of source-dataset (e.g., MPI) samples in a batch and `B_t` the number of target-dataset (e.g., H36M) samples. Define a **geometry consistency loss** on the shared 17-joint skeleton:

```
L_pose   = MSE(pred_3d, gt_3d)
L_bone   = Σ_j,k | ||pred_j - pred_k|| - ||gt_j - gt_k|| |
L_consistency = L_pose + λ_bone * L_bone + λ_cross * L_cross

where
L_cross = Σ_common_pairs || Δ_pred_MPI - Δ_pred_H36M ||
          (differences between corresponding joint pairs in a batch)
```

The bone-length term `L_bone` is already implemented in `motionflow_mv/losses/bone_length.py`. The cross-dataset term is new and acts as an auxiliary regularizer during mixed training.

### 2.5 Domain adaptation (optional but recommended)

For explicit cross-dataset transfer, reuse the existing `motionflow_mv/models/domain_adaptation_wrapper.py` (GRL + FiLM). It wraps the PP backbone and accepts `domain_labels`. The loss becomes:

```
L = L_consistency + λ_domain * L_domain + λ_mmd * L_mmd
```

The `DomainAdaptationWrapper` already supports this; only the data pipeline and an updated trainer are needed.

---

## 3. Code changes needed

### 3.1 New files

| Path | Purpose |
|------|---------|
| `motionflow_mv/data/skeleton_maps.py` | Common 17-joint names and per-dataset index maps |
| `motionflow_mv/data/canonical_17_dataset.py` | Load any canonical `.npz`, map to 17 joints, return `(x, y, K, R, t, dataset_id)` |
| `motionflow_mv/losses/cross_dataset_consistency.py` | Bone-length + cross-dataset geometry consistency loss |
| `experiments/train_webbridge_mixed_17joint_mpiinf3dhp.py` | Full mixed-dataset trainer |
| `experiments/eval_webbridge_mixed_17joint.py` | Unified evaluation over WebBridge datasets |
| `configs/benchmark_webbridge_mixed_17joint.yaml` | Manifest for `experiments/run_webbridge_benchmark.py` |

### 3.2 Files to modify (no changes to running experiments)

| Path | Change |
|------|--------|
| `motionflow_mv/data/mixed_dataset.py` | Add `aist`, `shelf`, `campus` to `DATASET_REGISTRY`; add skeleton-mapping hook in `__init__` |
| `motionflow_mv/data/__init__.py` | Export `skeleton_maps` and `canonical_17_dataset` |
| `experiments/eval_full_metrics.py` | Ensure 17-joint model can be evaluated on H36M and Shelf/Campus `.npz` files (already supported via `--source_n_views`) |
| `experiments/run_webbridge_benchmark.py` | Add `--dataset_registry` override so new datasets are listed |

### 3.3 Key functions / classes

- `motionflow_mv.data.skeleton_maps.map_to_common_17(npz_path, dataset_name)` — returns re-indexed arrays.
- `motionflow_mv.data.mixed_dataset.MixedDataset.__init__` — extended to accept `dataset_name` and apply mapping.
- `motionflow_mv.losses.cross_dataset_consistency.CrossDatasetConsistencyLoss.forward(pred, gt, dataset_ids)` — computes auxiliary loss.
- `experiments/train_webbridge_mixed_17joint_mpiinf3dhp.main` — orchestrates training.

---

## 4. Training & evaluation protocol

### 4.1 Datasets

**Training set (all canonical `.npz`):**

- MPI-INF-3DHP: `s_01_seq_01_v14_multiview_m.npz`, `s_01_seq_02_v14_multiview_m.npz`, `s_03_seq_01_v14_multiview_m.npz`, `s_03_seq_02_v14_multiview_m.npz`
- H36M (meters): `s_01_acts_{02-16}_multiview_m.npz`
- AIST++: split file from `data/webbridge/aistpp_canonical/`
- Shelf/Campus: `data/webbridge/shelf_campus/`

**Validation set:**

- MPI-INF-3DHP: `s_02_seq_01_v14_multiview_m.npz` (same as anchor)
- H36M: `s_09_acts_02_multiview_m.npz` (subject 9, action 2)
- Shelf/Campus: `Campus_Seq1/pseudogt_m.npz`

### 4.2 Loss

```
L = L_MSE + λ_bone * L_bone + λ_cross * L_cross + λ_pp * L_pp + λ_focal * L_focal
```

Recommended initial weights:

- `λ_bone = 0.1`
- `λ_cross = 0.05`
- `λ_pp = 0.2` (re-use anchor PP schedule)
- `λ_focal = 0.0` until smoke passes, then 0.01

### 4.3 Metrics

Run `experiments/eval_full_metrics.py` with the trained checkpoint on:

- MPI-INF-3DHP S2/Seq1: primary comparison vs. 8.75 mm anchor
- H36M S9/S11 full test set
- Shelf/Campus if ground truth available

Report:

- MPJPE (mm)
- PA-MPJPE (mm)
- PCK@50/100/150 mm
- PCK-AUC
- Per-joint MPJPE for failure analysis

### 4.4 Baseline to compare

Run the same evaluation on `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth` with `--model crossview_residual_pp --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128` on the exact same datasets. The mixed model must beat this anchor on MPI-INF-3DHP and also show meaningful cross-dataset accuracy on H36M.

### 4.5 Smoke test

Before full training, run 3 epochs on a tiny subset:

```bash
python experiments/train_webbridge_mixed_17joint_mpiinf3dhp.py \
  --mpi_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
  --h36m_train data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --batch_size 4 --epochs 3 \
  --smoke
```

Pass criteria:

- No NaNs or shape mismatches.
- At least one sample from each dataset appears in the first training batch.
- Validation MPJPE is finite and < 30 mm on the smoke split.

### 4.6 Full training command

```bash
python experiments/train_webbridge_mixed_17joint_mpiinf3dhp.py \
  --mpi_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
              data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
              data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
              data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --h36m_train data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
               data/webbridge/h36m_meters/s_01_acts_03_multiview_m.npz \
               ... \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
  --epochs 20 --batch_size 8 --train_samples 1000 \
  --output outputs/webbridge_mixed_17joint_pp.pth
```

---

## 5. Expected gains and risks

### 5.1 Expected gains

| Metric | Anchor | WebBridge mixed | Source of gain |
|--------|--------|-----------------|----------------|
| MPI-INF-3DHP MPJPE | 8.75 mm | **< 8.30 mm** target | Cross-dataset regularization, more diverse poses |
| H36M S9/S11 MPJPE | not optimized | **< 5 mm** | Direct H36M training signal (anchor is MPI-only) |
| Small-view robustness (3–5 views) | weak | measurable improvement | Shelf/Campus in training mix |
| Calibration robustness | good | similar or better | Shared intrinsics prior across datasets |

### 5.2 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Skeleton mapping ambiguity** (MPI 28 → 17) | Start with H36M + AIST only (both 17 joints); add MPI once mapping is validated. |
| **Negative transfer** from H36M studio rig hurting MPI accuracy | Use dataset-id FiLM in `DomainAdaptationWrapper`; reduce H36M sampling weight to 0.25× if MPI degrades. |
| **Scale mismatch** (H36M/AIST in meters vs. raw MPI) | WebBridge `.npz` files already use meters (`*_m.npz`); verify with `experiments/audit_webbridge_npz.py`. |
| **Shelf/Campus pseudo-GT noise** | Treat Shelf/Campus as auxiliary with lower loss weight; evaluate on clean H36M/MPI first. |
| **Training time** | Smoke first; full run is ~3–5 hours on RTX 4090, comparable to anchor. |
| **GPU occupied by Bayesian triangulation** | Smoke is CPU-friendly; queue full run after Bayesian tri finishes. |

---

## 6. Next steps for the follow-up implementer

1. **Create `motionflow_mv/data/skeleton_maps.py`.** Define `COMMON_17_JOINTS` and the four per-dataset maps. Add a unit test in `tests/test_skeleton_maps.py`.
2. **Extend `mixed_dataset.py`.** Add `aist`, `shelf`, `campus` to `DATASET_REGISTRY` and wire the mapping hook. Run a quick smoke to confirm padded batches mix correctly.
3. **Implement `CrossDatasetConsistencyLoss`.** Start with only the bone-length term; add the cross-dataset term after the bone-length smoke passes.
4. **Write `experiments/train_webbridge_mixed_17joint_mpiinf3dhp.py`.** Base it on `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`, but read multiple `--mpi_train`, `--h36m_train`, etc. arguments and use `build_mixed_dataloaders`.
5. **Run the smoke test.** Use `--smoke --epochs 3` and verify no NaNs, correct batch mixing, and finite validation error.
6. **Launch the full run** on the RTX 4090 after the Bayesian triangulation job finishes.
7. **Evaluate with `run_webbridge_benchmark.py`** using the new `configs/benchmark_webbridge_mixed_17joint.yaml` manifest and compare directly to the 8.75 mm anchor.

---

## 7. References to existing code

- Canonical loader: `motionflow_mv/data/webbridge_loader.py`
- Mixed dataset: `motionflow_mv/data/mixed_dataset.py`
- Temporal clip dataset: `motionflow_mv/data/temporal_clip_dataset.py`
- PP anchor model: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
- Domain adaptation wrapper: `motionflow_mv/models/domain_adaptation_wrapper.py`
- Eval harness: `experiments/eval_full_metrics.py`
- Benchmark runner: `experiments/run_webbridge_benchmark.py`
- Anchor comparison script: `scripts/compare_iter16_to_anchor.sh`
- Failure analysis: `docs/swarm_iter_next/failure_analysis_crossview_pp.md`
- Quality audit reports: `docs/swarm_iter_next/webbridge_npz_smoke_quality_report.md`, `docs/swarm_iter_next/webbridge_h36m_meters_quality_report.md`


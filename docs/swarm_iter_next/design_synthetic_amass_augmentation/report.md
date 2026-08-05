# Design Report: Synthetic SMPL/AMASS Augmentation (task_03)

**Author:** research swarm (task_03)  
**Date:** 2026-08-05  
**Scope:** Extend the synthetic multi-view data generator with SMPL/AMASS motion sampling, domain-randomized cameras, and a PyTorch Dataset wrapper.

## 1. Motivation

The project already had a basic synthetic generator in `motionflow_mv/data/synthetic_3d_dataset.py` (random floating skeletons) and a SMPL-based generator in `experiments/generate_synthetic_multiview_dataset.py`.  However, the SMPL generator had limited augmentation and no reusable library API.  Task_03 asks to **extend** the synthetic SMPL/AMASS augmentation so that:

1. Training scripts can consume synthetic data through a standard PyTorch `Dataset`.
2. Camera rigs can be domain-randomized to match H36M, MPI-INF-3DHP, or fully random distributions.
3. AMASS motion clips can be sampled when available, falling back to procedural motion otherwise.
4. The augmentation pipeline (noise, occlusion, outliers, mirror, scale jitter) is configurable and reusable.

## 2. Implementation

### 2.1 New / extended module: `motionflow_mv/data/synthetic_3d_dataset.py`

Added the following public components while keeping the original `make_cameras`, `generate_sequence`, and `generate_dataset` intact for backward compatibility.

| Class / Function | Purpose |
| --- | --- |
| `CameraRigSampler` | Sample camera rigs: `legacy`, `h36m`, `mpiinf3dhp`, `random`. |
| `MotionSampler` | Procedural Brownian poses + optional AMASS `*_poses.npz` sampling. |
| `AugmentConfig` | Configurable 2D augmentation parameters. |
| `augment_2d_keypoints` | Apply noise, occlusion, outliers, mirror, scale jitter. |
| `SMPLSequenceGenerator` | Generate one sequence: 2D, confidence, 3D GT, cameras, baseline. |
| `SyntheticMultiViewDataset` | PyTorch `Dataset` that yields `(x, y, K, R, t)` tuples. |
| `generate_synthetic_dataset` | High-level helper to write a canonical `.npz` file. |
| `project_points`, `triangulate_joints_baseline` | Reusable projection / baseline helpers. |

Key design decisions:

* **Graceful degradation:** `_check_smplx()` raises only when SMPL-specific code is invoked; the legacy API still works without `smplx`.
* **AMASS fallback:** If `amass_root` is provided, `MotionSampler` tries to load real motion clips; on any failure it falls back to procedural motion.
* **Camera diversity:** The `mpiinf3dhp` and `random` modes widen the camera distribution, which should improve synthetic-to-real transfer.
* **Baseline output:** Each sequence optionally returns a triangulated baseline, useful for training residual refinement models (e.g. `RayAttentionFusionModelTemporalResidual`).

### 2.2 Updated script: `experiments/generate_synthetic_multiview_dataset.py`

The script is now a thin CLI wrapper around the new `generate_synthetic_dataset` helper.  New CLI flags:

* `--camera_mode {h36m,mpiinf3dhp,legacy,random}`
* `--amass_root <path>`
* `--mirror_prob`, `--scale_jitter`
* `--seed`
* `--device`

Default behavior is unchanged (H36M-matched cameras, mm units, 500 sequences).

### 2.3 Smoke tests: `tests/test_synthetic_amass_augmentation.py`

Tests cover:

1. Legacy API shapes.
2. All four camera rig modes.
3. 2D augmentation pipeline.
4. Projection helper.
5. SMPL-based `.npz` generation (skipped if `smplx`/model absent).
6. PyTorch `Dataset` wrapper (skipped if `smplx`/model absent).

## 3. How to validate

Run the smoke tests (CPU, no training):

```bash
pytest tests/test_synthetic_amass_augmentation.py -v
```

Generate a small synthetic dataset for manual inspection:

```bash
python experiments/generate_synthetic_multiview_dataset.py \
    --n_sequences 5 --frames_per_seq 10 --n_views 4 \
    --camera_mode mpiinf3dhp \
    --output tmp/synthetic_smoke.npz
```

If AMASS is available:

```bash
python experiments/generate_synthetic_multiview_dataset.py \
    --amass_root data/amass \
    --n_sequences 20 --frames_per_seq 60 \
    --camera_mode h36m \
    --output outputs/synthetic_amass_dataset.npz
```

## 4. Expected impact

* **More realistic synthetic data:** Domain-randomized cameras and AMASS motions should narrow the synthetic-to-real gap.
* **Easier training pipelines:** `SyntheticMultiViewDataset` can be plugged into existing training scripts via the same `(x, y, K, R, t)` tuple format used by `temporal_clip_dataset.py`.
* **Faster iteration:** The new `.npz` generator supports multiple camera modes without editing code, useful for ablation studies.
* **No regressions:** Legacy functions are preserved.

## 5. Blockers / follow-up

* **Dependency:** `smplx` is not in `requirements.txt`.  If the team wants synthetic generation out-of-the-box, `smplx` should be added as an optional or test dependency.
* **AMASS data:** Real AMASS motion clips are not present in the repository; the implementation falls back to procedural motion when they are absent.
* **Skeleton mapping:** Currently hard-coded to 17 joints.  If future work needs H36M (17 joints) vs. MPI-INF-3DHP (17 joints) vs. COCO (17/18 joints), a joint remapping utility should be added.
* **Temporal augmentation:** Motion currently uses per-frame independent body poses (procedural) or contiguous AMASS clips.  Adding velocity smoothing / motion blur would further improve realism.

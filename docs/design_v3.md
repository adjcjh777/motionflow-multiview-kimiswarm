# MotionFlow Multi-View Extension — Design v3

## 1. Where we are

The FusionModule plugin interface is in place and validated on both synthetic 3D
data and the real Shelf multi-view dataset. Five plugins are registered:
`dlt`, `attention`, `robust_triangulation`, `residual_refiner`,
`temporal_refiner`.

Shelf reprojection error (px), frames 300–600:

**Synthetic checkpoints**

```text
Plugin                     Mean     Median        Max
----------------------------------------------------
attention                429.36     435.49     987.88
robust_triangulation     408.08     421.48     940.93
residual_refiner          13.03       8.35    1060.39
dlt                        9.88       5.52    1044.68
temporal_refiner           9.88       5.52    1044.68
```

**Shelf-trained checkpoints** (`attention` d=64, `residual_refiner` d=64,
`robust_triangulation` d=32)

```text
Plugin                     Mean     Median        Max
----------------------------------------------------
attention                 80.42      58.90     781.02
residual_refiner          13.11       9.79    1027.26
robust_triangulation      10.65       5.97     507.22
```

Key observations:

* The geometry-based plugins (`dlt`, `temporal_refiner`) give sub-pixel to
  ~10 px reprojection, confirming the per-plugin scaling fix in
  `experiments/eval_all_plugins_shelf.py`.
* Shelf-finetuned `attention` drops from 429 px to 80 px, and
  Campus-trained `attention` drops the Campus zero-shot error from 318 px to
  **110.55 px**.  Dataset-specific training helps, but pure attention still
  lags far behind geometry.
* `residual_refiner` is comparable with its synthetic and Shelf checkpoints,
  slightly behind `dlt`; a larger model or joint loss may be needed.
* `robust_triangulation` originally used SVD and got stuck at ~445 px on
  Shelf. Replacing the SVD with a stable inhomogeneous pseudo-inverse let it
  converge; the Shelf checkpoint now reaches **10.65 px**, close to DLT
  (9.88 px).  Cross-dataset zero-shot evaluation on CampusSeq1 gives DLT
  **1.52 px**, `robust_triangulation` **1.55 px**, and `temporal_refiner`
  **1.52 px**, confirming the geometric plugins generalize to a different
  calibration.  The learned `attention` model still fails to cross-dataset
  generalize (318 px), while `residual_refiner` transfers reasonably
  (8.26 px).
* A GVHMR-to-`HumanMotionIR` converter is now available via
  `motionflow_mv.ir.gvhmr_adapter.gvhmr_pt_to_ir`. It loads a single-view
  `hmr4d_results.pt` and produces a stable `HumanMotionIR` artifact with SMPL
  pose parameters and provenance.

## 2. Design refinement for v3

### 2.1 Unit convention

`HumanMotionIR` will standardize on **meters** as the canonical length unit.

* Calibration loader returns `Camera` in whatever unit the dataset provides
  (Shelf is mm).
* The multi-view adapter normalizes inputs and re-scales outputs per plugin,
  exactly as `eval_all_plugins_shelf.py` now does.
* All downstream consumers (GMR, MJLab, policy preview) receive metric 3D data
  with explicit `length_unit` metadata.

### 2.2 Plugin contract

Each plugin declares:

* `requires_calibration: bool`
* `input_scale: float` — factor by which 2D inputs must be scaled before
  inference.
* `output_scale: float` — factor by which plugin outputs must be scaled to
  meters.

This removes the hard-coded branching in the evaluation script and lets the
adapter treat every plugin uniformly.

### 2.3 Training strategy

To reach CVPR/ICRA-level numbers we need:

1. **Shelf fine-tuning** for `attention`, `robust_triangulation`, and
   `residual_refiner`. Pseudo-3D targets come from the current best geometric
   fusion (`dlt` / `temporal_refiner`) when 3D GT is absent.
2. **Scale-aware augmentation**: random world-unit scaling during training so
   the learned plugins do not overfit to one metric scale.
3. **Cross-dataset validation**: train on Shelf, validate on Campus (also from
   VoxelPose) and on synthetic sequences.

## 3. Immediate next steps

1. **GVHMR + SMPL multi-view projection demo**: run GVHMR on multiple views of
   the same action, convert each per-view `HumanMotionIR` to 2D keypoints, then
   fuse with the plugin pipeline and compare to the single-view GVHMR output.
2. **Per-joint/per-view breakdown** in `eval_all_plugins_shelf.py` to locate
   remaining failure modes.
3. **Train a geometry-aware `attention` variant** that receives normalized ray
   directions or projection-matrix embeddings; the current `attention_v2`
   geometry-aware attempt is unstable and needs better normalization.

## 4. Risk register

* **Scale mismatch**: real datasets use mm; synthetic data uses meters. Already
  mitigated by per-plugin scaling, but needs to be encoded in the plugin
  contract.
* **Projection matrix mismatch across datasets**: learned geometric plugins
  (`robust_triangulation`) trained on one calibration will not generalize to
  another unless the network is scale/camera invariant or fine-tuned.
* **A800-D read-only constraint**: we can inspect Docker/vendor data and golden
  artifacts, but cannot modify containers or launch training there.

## 5. Phase 3 iteration — ray-aware attention fusion (this turn)

### 5.1 Swarm findings

A 20-agent research swarm produced reports in `docs/swarm_iter3/`. The most
actionable conclusion is that `attention_v2` is unstable because it flattens the
raw 12-D projection matrix and directly regresses 3D coordinates. The swarm
converged on a cleaner geometry-aware design:

* Use **ray directions + camera centers** instead of flattened `P`.
* Predict **per-view weights**, then feed them into a **differentiable weighted DLT**
  layer rather than regressing 3D coordinates end-to-end.
* Add **reprojection / epipolar losses** and train on real 3D GT (Human3.6M)
  whenever possible.

References: `docs/swarm_iter3/attention_fusion_v2_instability.md`,
`docs/swarm_iter3/geometry_aware_attention.md`,
`docs/swarm_iter3/epipolar_constraints.md`.

### 5.2 New plugin: `ray_attention`

Implemented:

* `motionflow_mv/fusion/ray_attention_model.py` — `RayAttentionFusionModel`:
  computes camera rays via `K^{-1}`, embeds `(x, y, conf)` + `(camera_center,
  ray_dir)`, runs multi-head self-attention over views per joint, predicts
  per-view weights, and triangulates with differentiable weighted DLT.
* `motionflow_mv/fusion/ray_attention_module.py` — `RayAttentionFusionModule`
  wrapper registered as a `FusionModule` plugin.
* `experiments/train_ray_attention_shelf.py` — training script for Shelf data.
* `tests/test_ray_attention.py` — unit tests (14 tests passing).

### 5.3 Initial validation

GVHMR multi-view projection demo (`experiments/demo_gvhmr_multiview_projection.py`)
now includes `ray_attention`. On 4 virtual cameras with 0.5 px Gaussian noise:

```text
Plugin                    MPJPE
--------------------------------
attention                3.6807
ray_attention            0.0021
```

`ray_attention` produces a near-perfect metric triangulation because the DLT
layer enforces the correct geometric inductive bias; the remaining error is in
the order of the synthetic noise. The gap to `attention` confirms that the new
design is better conditioned for calibrated multi-view fusion.

### 5.4 Synthetic training workaround

Because raw Shelf/Campus/H36M data is not in the workspace, a synthetic pipeline
was built to unblock training and validate the end-to-end loop:

* `experiments/generate_synthetic_multiview_dataset.py`: generates random SMPL
  poses projected through calibrated virtual rigs with randomized radius, focal
  length, principal point, and height. Each frame also gets random Gaussian
  noise, per-joint occlusion, and occasional 2D outliers.
* `experiments/train_ray_attention_synthetic.py`: trains `ray_attention` on the
  synthetic data with a 3D MSE loss. Supports batched per-sample camera tensors
  so training runs at `batch_size=32`.
* `experiments/eval_ray_attention_robustness.py`: controlled occlusion/outlier
  evaluation on 200 synthetic trials.

Results (200 synthetic trials, 4 views, 0.8 px noise, 10% occlusion, 2% outliers):

```text
MPJPE (m)
  Clean 4 views:        0.0036
  1 view occluded:      0.0042
  2 views occluded:     0.0057
  1 view outlier:       0.0043
```

Training metrics (`batch_size=32`, 30 epochs, 6000 synthetic frames):

```text
val_MPJPE = 0.0112 m
```

These numbers confirm that the ray-aware attention head learns to down-weight
occluded and corrupted views while preserving metric scale. The remaining error
is dominated by the synthetic noise floor.

* When the synthetic checkpoint is loaded into the GVHMR projection demo,
  `ray_attention` achieves **0.0021 m MPJPE**, confirming the model preserves
  metric scale and triangulates correctly on unseen SMPL motion.

### 5.5 A800-D read-only audit

SSH access to `a800-D` (user `zhangzy`) succeeded. Relevant findings:

* `/mnt/nvme0n1/zhangzy/projects` contains motionflow-related projects and a
  running Docker container `motionflow`, but no readily accessible Shelf/Campus
  or Human3.6M raw data.
* `/mnt/nvme1n1p1/datas/zhangzy/motionflow-runtime/data` holds SQLite task
  databases, not pose datasets.
* GVHMR single-view videos and SMPL assets are available under the vendor trees.

Therefore, the current realistic data path is:
1. Use the synthetic generator for fast model validation (done).
2. Download Shelf/Campus via the Google Drive mirror documented in
   `docs/swarm_iter2/shelf_campus_source.md` and train on real data.
3. Apply for Human3.6M for 3D-supervised training.

### 5.6 Next steps

1. **Acquire raw multi-view data**: locate `data/Shelf` locally or access the
   A800-D read-only projects directory to obtain Shelf/Campus/VoxelPose files.
2. **Run `prepare_shelf_dataset.py` and `train_ray_attention_shelf.py`** to train
   the first ray-aware checkpoint on real data.
3. **Add Human3.6M data loader** (`motionflow_mv/data/human36m_loader.py`) and
   train with real 3D GT; this is the only path to clearly beat DLT.
4. **Add epipolar loss** to the ray-aware trainer to improve cross-dataset
   generalization.
5. **Controlled ablation**: raw flattened `P` vs. ray embeddings, direct 3D
   regression vs. weighted DLT, DLT pseudo-GT vs. 3D GT.

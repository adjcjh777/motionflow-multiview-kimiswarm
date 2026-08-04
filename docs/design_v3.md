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

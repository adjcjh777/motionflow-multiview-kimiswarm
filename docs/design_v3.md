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

**Shelf-trained checkpoints** (`attention` d=64, `residual_refiner` d=64)

```text
Plugin                     Mean     Median        Max
----------------------------------------------------
attention                 80.42      58.90     781.02
residual_refiner          13.11       9.79    1027.26
```

Key observations:

* The geometry-based plugins (`dlt`, `temporal_refiner`) give sub-pixel to
  ~10 px reprojection, confirming the per-plugin scaling fix in
  `experiments/eval_all_plugins_shelf.py`.
* Shelf-finetuned `attention` drops from 429 px to 80 px once the architecture
  and scale are matched, showing the value of real-data fine-tuning.
* `residual_refiner` is comparable with its synthetic and Shelf checkpoints,
  slightly behind `dlt`; a larger model or joint loss may be needed.
* `robust_triangulation` was fine-tuned on Shelf but did not improve beyond the
  synthetic baseline (still ~445 px). The differentiable SVD layer appears to
  get stuck near a degenerate solution, so a reparameterization or
  reprojection-only training with better initialization is needed.

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

1. **Robust triangulation redesign**: replace the differentiable SVD with a
   stable weighted pseudo-inverse or reparameterize the per-view weights with a
   softmax + entropy regularizer, then retrain on Shelf.
2. **Add a Campus dataset loader** and cross-dataset validation (train on Shelf,
   validate on Campus) to test generalization beyond Shelf calibration.
3. **GVHMR + SMPL multi-view projection demo**: use read-only A800-D
   Docker/vendor data to validate the `HumanMotionIR` end-to-end path.
4. **Per-joint/per-view breakdown** in `eval_all_plugins_shelf.py` to locate
   remaining failure modes.

## 4. Risk register

* **Scale mismatch**: real datasets use mm; synthetic data uses meters. Already
  mitigated by per-plugin scaling, but needs to be encoded in the plugin
  contract.
* **Projection matrix mismatch across datasets**: learned geometric plugins
  (`robust_triangulation`) trained on one calibration will not generalize to
  another unless the network is scale/camera invariant or fine-tuned.
* **A800-D read-only constraint**: we can inspect Docker/vendor data and golden
  artifacts, but cannot modify containers or launch training there.

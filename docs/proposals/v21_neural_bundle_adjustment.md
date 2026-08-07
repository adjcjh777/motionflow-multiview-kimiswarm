# v21: Neural Bundle-Adjustment Layer

## Motivation

Classical bundle adjustment (BA) is the gold standard for refining 3D structure and cameras from multi-view observations, but it is:

* **Brittle at initialization** — full free-form BA easily diverges when the initial pose/calibration is noisy.
* **Computationally heavy** — many Gauss-Newton iterations are needed.
* **Not easily differentiable** — standard solvers break autograd.

The existing `DifferentiableBundleAdjustment` layer in the codebase only refines the **3D skeleton** while keeping cameras fixed. v21 extends this idea to a **neural bundle-adjustment layer** that jointly refines both 3D joints and calibrated cameras in a lightweight, fully differentiable block.

## Design

### Module: `NeuralBundleAdjustment`

Location: `motionflow_mv/fusion/neural_bundle_adjustment_v21.py`

The layer alternates between two steps for a small number of iterations (default `n_iters=2`):

1. **Camera-correction step (neural)** — a small MLP looks at per-view reprojection statistics and predicts bounded updates to the intrinsic and extrinsic parameters.
2. **Structure step (analytic Gauss-Newton)** — using the updated cameras, a single damped Gauss-Newton step refines every 3D joint by minimizing the weighted reprojection residual.

### Inputs / Outputs

**Forward signature**

```python
X_ref, K_ref, R_ref, t_ref = nba(
    X,           # (B, T, J, 3) or (B, J, 3)   initial 3D joints
    points_2d,   # (B, T, V, J, 2) or (B, V, J, 2) observed 2D keypoints
    K,           # (B, T, V, 3, 3) or (B, V, 3, 3) intrinsics
    R,           # (B, T, V, 3, 3) or (B, V, 3, 3) rotations
    t,           # (B, T, V, 3) or (B, V, 3) translations
    weights,     # optional per-view/joint weights
)
```

All outputs have the same shapes as their corresponding inputs.

### Camera-Correction Head (`_CameraCorrectionHead`)

The head builds a per-camera descriptor from:

* Mean and standard deviation of the reprojection residual (2 + 2 dims)
* Current intrinsics: `fx, fy, cx, cy, skew` (5 dims)
* Flattened rotation matrix (9 dims)
* Translation vector (3 dims)
* Total per-view weight proxy (1 dim)

total `in_dim = 22`.

The MLP outputs a 9-D correction vector:

* `df` — multiplicative focal-length scale
* `dpp` — principal-point offset
* `daxis` — axis-angle rotation update (bounded, applied via Rodrigues)
* `dt` — translation update

All updates are `tanh`-bounded so the layer stays near identity and is safe to stack.

### Structure Update (`_update_points`)

Reuses the analytic projective Jacobian from `differentiable_bundle_adjustment.py`.
For each joint it solves

```
(J^T W J + λ I) ΔX = J^T W r
```

with per-view weights normalized over views. The update is clamped to keep the optimization stable.

## Integration

The layer is intended to be used as a post-processing refinement block inside larger fusion models, e.g.:

```python
from motionflow_mv.fusion.neural_bundle_adjustment_v21 import NeuralBundleAdjustment

nba = NeuralBundleAdjustment(n_iters=2, camera_hidden=64)
X_ref, K_ref, R_ref, t_ref = nba(pred_3d, points_2d, K, R, t, weights)
```

Because it is fully differentiable and operates on tensors, it can be inserted after any triangulation head and trained end-to-end with the usual 3D pose losses.

## Test Coverage

`tests/test_neural_bundle_adjustment_v21.py` covers:

* Forward pass shape consistency with and without a temporal dimension.
* Rotation matrices remain orthogonal after refinement.
* Backward pass — gradients flow through both the neural camera head and the analytic point update.

Run with:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_neural_bundle_adjustment_v21.py -v
```

## Future Work

* Add confidence-aware masking of missing views.
* Use a deeper camera-correction network (e.g. transformer over joints) when strong calibration errors are expected.
* Couple the point and camera updates in a single linear system for true joint optimization.

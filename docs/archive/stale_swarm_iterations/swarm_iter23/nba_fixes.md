# Neural Bundle Adjustment (v21) — Concrete Fixes

**Scope:** `motionflow_mv/fusion/neural_bundle_adjustment_v21.py`  
**Context:** v21 regressed to 128.27 mm and was stopped. v23 currently omits neural BA entirely. Goal: make the neural camera-correction block safe and beneficial so it can be re-introduced in a later iteration without destabilizing training.

---

## 1. Refine structure *before* touching the cameras

**Observation:** The alternating loop updates cameras first, then points (`neural_bundle_adjustment_v21.py:304-306`):

```python
for _ in range(self.n_iters):
    K, R, t = self._update_cameras(X, points_2d, K, R, t, weights)
    X = self._update_points(X, points_2d, K, R, t, weights)
```

The neural camera-correction head therefore sees reprojection residuals that are polluted by the *initial* noisy 3D skeleton. It has no chance to disentangle point error from camera error, so it can easily move perfectly good cameras to explain bad points.

**Proposal:** Swap the order and, optionally, insert one structure-only warm-up step:

```python
for _ in range(self.n_iters):
    X = self._update_points(X, points_2d, K, R, t, weights)
    K, R, t = self._update_cameras(X, points_2d, K, R, t, weights)
```

Add a boolean `warm_start_structure: bool = True`; when true, run a single `_update_points` before the first camera update.

**Why:** After a point update, the residual is dominated by camera/model error, giving the camera head a cleaner signal. This is the standard BA ordering (structure → cameras).

**Risk / mitigation:** The point update may be noisy in the first iteration; keep `max_point_update` small (≤ 0.05 m) and the LM `damping` ≥ 1.0. Run a quick smoke test with the existing `tests/test_neural_bundle_adjustment_v21.py`.

**Files to touch:** `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` (`forward` loop).

---

## 2. Replace the 9-element rotation matrix in the camera descriptor with a compact, constrained representation

**Observation:** `_camera_descriptor` flattens the full 3×3 rotation matrix into 9 numbers (`neural_bundle_adjustment_v21.py:88`). Rotation has only 3 DOF; the remaining 6 entries are redundant and subject to orthogonality constraints that the MLP must learn implicitly. This makes the correction head harder to train and more likely to overfit to residual patterns that are actually point-noise artifacts.

**Proposal:** Replace the rotation descriptor with the axis-angle vector of `R` (or the 6D Gram-Schmidt representation). For example:

```python
# Axis-angle of the view rotation (B, T, V, 3).
rot_aa = _rotation_matrix_to_axis_angle(R)
```

Then change the descriptor to:

```python
return torch.cat([mean_res, std_res, intr, rot_aa, trans, weight_sum], dim=-1)
```

Update `in_dim` from 22 to 16 (2 + 2 + 5 + 3 + 3 + 1). Keep the output update still parameterized by axis-angle for consistency.

**Why:** A 3-DOF rotation descriptor removes redundancy and gives the MLP a better-conditioned camera representation. This is especially helpful early in training when the head must learn to leave cameras alone (identity init).

**Risk / mitigation:** Converting R → axis-angle is multi-valued; use `torch.matrix_to_euler_angles` or a stable `rotation_matrix_to_axis_angle` helper. Add a unit test that checks the descriptor dimension and that zero rotation maps to a zero axis-angle.

**Files to touch:** `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` (`_camera_descriptor`, `NeuralBundleAdjustment.__init__` for `in_dim`).

---

## 3. Detach the structure update from the camera-head gradient and gate corrections by residual improvement

**Observation:** Inside the loop, the camera head’s gradient flows through the freshly updated points (`X = self._update_points(...)` → `_update_cameras(X, ...)`). The camera head can therefore “cheat” by driving gradients through the analytic point update, which amplifies instability. There is also no guard against a camera update that *increases* reprojection error, which is exactly the failure mode seen in v21.

**Proposal:**

1. Detach the point update before the camera head sees it:

```python
for _ in range(self.n_iters):
    X = self._update_points(X, points_2d, K, R, t, weights)
    K, R, t = self._update_cameras(X.detach(), points_2d, K, R, t, weights)
```

2. Add a residual-improvement gate: compute the per-view mean reprojection error before and after the camera update, and only accept the new cameras if the error does not increase:

```python
def _mean_reproj_error(X, points_2d, K, R, t, weights):
    residual, _, valid = _project_and_jacobian(X, points_2d, K, R, t)
    err = (residual ** 2).sum(-1)  # (B, T, V, J)
    return (err * weights * valid.float()).sum() / (weights * valid.float()).sum().clamp_min(1e-6)
```

```python
err_before = _mean_reproj_error(X, points_2d, K, R, t, weights)
K_new, R_new, t_new = self._update_cameras(X.detach(), points_2d, K, R, t, weights)
err_after = _mean_reproj_error(X, points_2d, K_new, R_new, t_new, weights)

use_new = (err_after < err_before + self.camera_update_tol).float()
# Broadcast use_new to (B, T, V, 1, 1) for K/R and (B, T, V, 1) for t.
K = use_new[..., None, None] * K_new + (1 - use_new[..., None, None]) * K
R = use_new[..., None, None] * R_new + (1 - use_new[..., None, None]) * R
t = use_new[..., None] * t_new + (1 - use_new[..., None]) * t
```

Make this gating optional via `gate_camera_update: bool = True` so training can still backprop through the MLP.

**Why:** Detaching breaks the dangerous gradient path through the analytic point update. The residual gate guarantees that a camera update cannot make reprojection worse, which directly prevents the v21-style regression.

**Risk / mitigation:** Gating is piecewise-constant; it is fine for inference and for stabilizing early training. For end-to-end gradient, keep the gate off (`gate_camera_update=False`) once the head is warm, or use a soft gate (`sigmoid(err_before - err_after)`) instead of hard masking.

**Files to touch:** `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` (`forward`, new `_mean_reproj_error` helper); add corresponding unit tests in `tests/test_neural_bundle_adjustment_v21.py`.

---

## Validation plan

Apply the fixes incrementally and measure:

1. **Smoke test:** Run `tests/test_neural_bundle_adjustment_v21.py`; all shape/orthogonality/backward tests should still pass.
2. **Synthetic sanity:** Generate a calibrated multi-view sequence with known ground-truth 3D joints and small camera perturbations. After one NBA forward pass, reprojection error should decrease and camera corrections should stay within their bounds.
3. **End-to-end guard:** Plug the updated block back into `omniview_fusion_v5.py` on a 1-epoch local smoke run. Compare `val_MPJPE` and `val_Reproj` against the v18 baseline. If the updated block still raises error or MPJPE by > 5 mm, keep it disabled for v23.

## Quick-win prioritization

1. **Highest impact / lowest risk:** Item 1 (structure-first ordering) — a one-line reorder that removes the biggest source of camera-head confusion.
2. **Best training stability:** Item 3 (detach + residual gate) — directly protects against the 128 mm regression by construction.
3. **Best representational hygiene:** Item 2 (compact rotation descriptor) — improves the MLP’s learning problem and slightly reduces parameter count.

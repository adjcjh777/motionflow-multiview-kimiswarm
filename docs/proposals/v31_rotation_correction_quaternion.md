# v31: Quaternion Rotation Correction for Multi-View Extrinsics

## Problem Statement

v25–v30 rely on `RotationCorrectionHead` (`motionflow_mv/fusion/rotation_correction.py`) to compensate for small camera-orientation errors before triangulation. The current head predicts a bounded axis-angle vector `r ∈ ℝ³` and converts it to an SO(3) matrix with `torch.linalg.matrix_exp`. While this is mathematically clean, the matrix exponential is relatively expensive, and its gradients can be numerically noisy for very small residual angles. More importantly, the axis-angle representation carries no natural “identity-at-init” normalization that is cheap to enforce—bounded by `tanh`, but still requiring an exponential map.

We want a drop-in alternative that is cheaper, numerically friendlier, and keeps the same bounded, identity-at-init semantics, while integrating cleanly with the v30 hardened hierarchical encoder.

## Concrete Proposed Change

Introduce `QuaternionRotationCorrectionHead` in a new module `motionflow_mv/fusion/rotation_correction_quaternion.py`. It is API-compatible with `RotationCorrectionHead`:

```python
class QuaternionRotationCorrectionHead(nn.Module):
    def forward(self, feat, R) -> Tuple[Tensor, Tensor]:
        # returns (R_corrected, delta_R)
```

The new head predicts a 3-D vector `v = tanh(mlp(feat)) * max_rot_rad / 2`, where `max_rot_deg` is the same bound used today (default 2°). A unit quaternion is formed by fixing the scalar part to the positive hemisphere:

```
q = [sqrt(1 - ||v||²), v_x, v_y, v_z]
```

This guarantees `|v| ≤ 1` (via `tanh`), so the square root is always real. The quaternion is converted to a 3×3 rotation matrix with the standard Hamilton product formula. The residual is applied as `R_corrected = delta_R @ R`, identical to the current head.

Key design points:

- **Identity at init:** the final linear layer is zero-initialized, so `v = 0`, `q = [1, 0, 0, 0]`, and `delta_R = I`.
- **No matrix exponential:** quaternion → matrix is a few arithmetic ops, cheaper than `matrix_exp` and stable for tiny angles.
- **Same bound:** each axis-angle component is still bounded by `max_rot_deg`, preserving the safety of the current head.
- **Drop-in:** replace `rotation_correction_head = RotationCorrectionHead(...)` with `rotation_correction_head = QuaternionRotationCorrectionHead(...)` inside `OmniMultiViewFusionV5` when a new flag `use_quaternion_rotation_correction` is enabled.

Training run (local smoke):
- Base: v30 hardened hierarchical encoder + physical loss.
- Disable TTE (broken, per current constraints).
- Physical loss warmup: `--v29_physical_loss_warmup_epochs 2`.
- Add `--use_quaternion_rotation_correction` and keep `--use_rotation_correction true` for the base path.

## Expected Impact on val_MPJPE / Overfitting

- **val_MPJPE:** small positive (<1 mm) or neutral. The representation is more efficient but the representational capacity is unchanged, so gains are expected to be modest. The main upside is a cleaner gradient path during the first epoch, which may help the v30 encoder reach its best val_MPJPE faster.
- **Overfitting:** should not increase. The number of learnable parameters is identical (same MLP outputting 3 values). The fixed positive-scalar quaternion avoids the `q ↔ -q` double-cover ambiguity near the identity because `q = [1, 0, 0, 0]` is unique.
- **Speed:** a small per-batch speed-up because the matrix exponential is replaced by closed-form quaternion-to-matrix arithmetic.

## Main Risk

- **Quaternion double cover:** although the scalar-part construction pins us to the positive hemisphere near the identity, if `||v||` ever approaches 1 the opposite quaternion would represent the same rotation. In practice `max_rot_deg = 2°` keeps `||v||` tiny, so this is not a concern, but it should be validated with a smoke test.
- **Numerical drift:** repeated quaternion-to-matrix conversion could accumulate tiny orthogonality errors over many layers. We mitigate this by using the closed-form conversion and only applying the correction once before triangulation.

## Suggested Smoke Command

```bash
bash scripts/launch_v31_rotation_correction_quaternion_local4090.sh
```

This runs a 3-epoch smoke on the local RTX 4090 with the v30 base + quaternion rotation correction, without TTE and with physical-loss warmup, so we can verify both convergence and SO(3) validity before any full-scale A800 run.

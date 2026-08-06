# Curriculum Checkpoint Interim Robustness Results

**Checkpoint:** `outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth` (interim, training continues)

| Condition | MPJPE (mm) | PA-MPJPE (mm) | root_rel | velocity | PCK@50 | PCK@100 | PCK@150 | AUC |
|-----------|------------|-----------------|----------|----------|--------|---------|---------|-----|
| clean     | 10.76      | 5.42            | 5.15     | 1.04     | 1.00   | 1.00    | 1.00    | 0.928 |
| rot_0.5   | 26.92      | 9.58            | 7.30     | 3.20     | 0.92   | 1.00    | 1.00    | 0.821 |
| trans_5   | 12.20      | 5.57            | 5.24     | 1.37     | 1.00   | 1.00    | 1.00    | 0.919 |
| focal_1%  | 10.66      | 5.65            | 5.37     | 1.04     | 1.00   | 1.00    | 1.00    | 0.929 |
| pp_10px   | 2070.33    | 451.17          | 420.24   | 38.66    | 0.00   | 0.00    | 0.00    | 0.000 |

## Observations

- Clean accuracy is competitive at 10.76 mm / 5.42 PA-MPJPE.
- Translation and focal-length perturbations are handled well.
- Rotation remains a weakness (26.92 mm at ±0.5°).
- Principal-point shifts (±10 px) remain catastrophic, indicating the PP correction head has not yet learned to recover large offsets for this interim checkpoint.

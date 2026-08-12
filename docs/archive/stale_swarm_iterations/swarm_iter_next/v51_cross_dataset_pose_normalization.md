# v51 Cross-Dataset Pose Normalization (CDPN)

## Focus-area statement

H36M, MPI, 3DPW and WebBridge define human poses with different global scales, root heights, and bone-length distributions. While v48 domain generalization adapts the feature/representation level, it does not explicitly canonicalize the *geometry* of the 3-D pose. v51 introduces a lightweight cross-dataset pose normalization module that learns per-dataset scale/translation/bone-length transformations, lets the v46/v47/v48 sparse-view and temporal modules operate in a shared canonical skeleton space, and then maps back to the original dataset space for the supervised loss.

## Architecture

- **Module**: `CrossDatasetPoseNormalizationV51` → `motionflow_mv/fusion/cross_dataset_pose_normalization_v51.py`
- **Input**: triangulated 3-D pose `P ∈ R^{J×3}` and domain ID `d`.
- **Per-dataset transform**: learnable affine normalization per dataset,
  `P_canonical = s_d * P + t_d`.
  - `s_d` is a scalar global scale (clamped to `[0.8, 1.2]`), initialized to `1.0`.
  - `t_d ∈ R^3` is a learnable root translation, initialized to `0`.
- **Optional bone-length re-centering**: when enabled, a tiny MLP (`1 layer`) predicts per-domain bone-length multipliers that are applied to the canonical pose to match a shared canonical skeleton; identity at init preserves the baseline.
- **Integration point**: insert after the first triangulation in `omniview_fusion_v5.py` and before the v46 reliability head and v50 self-evolution feedback head. Downstream modules consume `P_canonical`; the supervised loss is computed on `P_original = (P_canonical - t_d) / s_d`.

## New config flags

| Flag | Type | Default |
|---|---|---|
| `use_v51_cross_dataset_pose_normalization` | bool | `False` |
| `v51_cdpn_use_scale` | bool | `True` |
| `v51_cdpn_use_translation` | bool | `True` |
| `v51_cdpn_learn_bone_length` | bool | `True` |
| `v51_cdpn_identity_init` | bool | `True` |
| `v51_cdpn_scale_min` | float | `0.8` |
| `v51_cdpn_scale_max` | float | `1.2` |
| `v51_cdpn_bone_reg_weight` | float | `0.001` |
| `loss.v51_cdpn_loss_weight` | float | `0.01` |

## Loss term

`L_cdpn = loss.v51_cdpn_loss_weight * [ MSE(P_canonical, teacher_P_canonical) + v51_cdpn_bone_reg_weight * Σ_j (||b_j(P_canonical)|| - b_j^canon)^2 ]`

- The first term is a consistency loss against a momentum-updated teacher canonical pose (or the v46/v48 baseline when no teacher is available).
- The second term regularizes canonical bone lengths toward a shared set `b^canon` learned online as the dataset-wise median.

## Evaluation metric

- Primary: `val_MPJPE@full` and `MPJPE@k` for `k = 2,3,4` on the in-domain val set.
- Cross-domain: per-domain `MPJPE@k` on H36M / MPI / 3DPW actual; 3DPW actual `MPJPE@2` and `MPJPE@3`.
- Diagnostic: canonical bone-length alignment `ΔBL = (1/J) Σ_j std_d | b_j(P_canonical,d) - b_j^canon |`; target < 5 mm.

## Expected MPJPE impact

- 3DPW actual `MPJPE@2`: **−3 to −5 mm**
- 3DPW actual `MPJPE@3`: **−2 to −3 mm**
- In-domain full-view `val_MPJPE`: **−0.8 to −1.5 mm**
- The biggest gains should appear in the sparse-view (`MPJPE@2`) and cross-domain regimes where scale/bone-length mismatch currently amplifies triangulation error.

## Main risk and mitigation

**Risk**: Over-normalization can collapse scale information that the triangulation and v46 reliability head rely on, causing `MPJPE@full` to regress or NaNs during inversion.

**Mitigation**:
1. Identity-at-init (`s_d=1`, `t_d=0`).
2. Clamp `s_d` to a narrow range `[0.8, 1.2]`.
3. Use a low auxiliary weight (`0.01`) and a teacher/baseline consistency term instead of forcing a fixed canonical target.
4. Smoke-test with `v51_cdpn_use_scale=False` first, then enable scale once translation-only mode is stable.

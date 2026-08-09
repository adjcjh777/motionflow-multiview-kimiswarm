# v53 Camera Noise Correction (CNC)

**Author:** design-swarm agent (v53 module proposal)  
**Builds on:** v52 Uncertainty-Weighted Triangulation (UWT)  
**Tracking issue:** #184 (proposed)  

## 1. Motivation

The current MotionFlow-MultiView pipeline (v25 → v52) treats 2-D keypoints as "clean" observations and relies on v52 to learn per-view/joint precision weights. In practice, multi-view setups suffer from calibration drift, rolling-shutter time skew, lens distortion residuals, and detector jitter that are **shared across all joints in a view**. These errors are not random Gaussian noise: they create coherent 2-D displacements that biased triangulation cannot resolve with per-joint weights alone.

The paper story is: *multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline*. v53 explicitly addresses the **calibration** part of that story by learning a differentiable camera-noise correction that refines the 2-D observations before they enter the v52 triangulation head. Because the correction is gated and zero-initialized, it is warm-startable from any v52 checkpoint.

## 2. Architecture

`CameraNoiseCorrectionV53` sits immediately **before** `UncertaintyWeightedTriangulationV52` inside `OmniMultiViewFusionV5.forward`. It predicts a per-view (optionally per-joint) 2-D correction in normalized image coordinates, applies it to `points_2d`, and passes the corrected points into v52.

### 2.1 Inputs and outputs

```text
features       : (B, T, V, J, d)   feature tokens from the cross-view encoder
points_2d      : (B, T, V, J, 2)   raw 2-D keypoints in pixel coordinates
K, R, t        : (B, T, V, 3, 3), (B, T, V, 3, 3), (B, T, V, 3)
pred_3d_init   : (B, T, J, 3)      initial triangulated estimate from the v25 block
view_mask      : (B, T, V)         bool, True = view valid

outputs:
  points_2d_corr : (B, T, V, J, 2)
  cnc_loss       : scalar auxiliary loss
  delta_norm     : (B, T, V, J)   correction magnitude for logging
```

### 2.2 Correction model

1. **Feature aggregation.** Pool per-view statistics from `features` and the reprojection residual of `pred_3d_init`:

   \[
   f_v = \text{MLP}\big[\underbrace{\frac{1}{J}\sum_j f_{v,j}}_{\text{mean token}},\;
         \underbrace{\text{std}_j(f_{v,j})}_{\text{token std}},\;
         \underbrace{\frac{1}{J}\sum_j r_{v,j}}_{\text{mean reproj. residual}}\big]
   \]

   where \(f_{v,j} \in \mathbb{R}^d\) and \(r_{v,j}\) is the v52 reprojection residual norm.

2. **Per-view affine correction.** A small MLP outputs a 6-parameter affine in the normalized image plane:

   \[
   \theta_v = (a_{11}, a_{12}, t_x, a_{21}, a_{22}, t_y) \in \mathbb{R}^6
   \]

   organized as

   \[
   A_v = \begin{bmatrix}
   1 + a_{11} & a_{12} & t_x \\
   a_{21} & 1 + a_{22} & t_y \\
   0 & 0 & 1
   \end{bmatrix}.
   \]

3. **Apply correction.** Convert the pixel keypoint to a normalized ray, apply the affine, and reproject:

   \[
   \tilde{x}_{v,j} = K_v^{-1}\, [u_{v,j},\; 1]^T, \qquad
   \tilde{x}'_{v,j} = A_v \, \tilde{x}_{v,j}, \qquad
   u'_{v,j} = \Pi(K_v \, \tilde{x}'_{v,j})
   \]

   where \(\Pi(\cdot)\) is perspective division. At init, \(\theta_v = 0\), so \(A_v = I\) and \(u'_{v,j} = u_{v,j}\).

4. **Gating.** A scalar gate is predicted per view (or global) and multiplied by the correction. The final corrected point is:

   \[
   u^\text{corr}_{v,j} = u_{v,j} + g_v \cdot (u'_{v,j} - u_{v,j})
   \]

   with \(g_v\) initialized to 0. This makes the module strictly identity-at-init and warm-startable.

5. **Loss.** An lightweight auxiliary loss keeps corrections bounded and physically consistent:

   \[
   \mathcal{L}_\text{cnc} =
   \lambda_1 \underbrace{\mathbb{E}[\|u^\text{corr} - u\|_2]}_{\text{correction magnitude}} +
   \lambda_2 \underbrace{\mathbb{E}[\|r^\text{corr} - r^\text{init}\|_2]}_{\text{reprojection change}}
   \]

   where \(r^\text{corr}\) is the residual after applying the corrected points but **before** v52 refinement, which prevents the module from hiding behind v52’s weights.

## 3. Config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v53_camera_noise_correction` | bool | `False` | Enable the v53 head. |
| `v53_cnc_hidden` | int | `64` | Hidden dim of the per-view MLP. |
| `v53_cnc_n_layers` | int | `2` | MLP layers. |
| `v53_cnc_per_joint_residual` | bool | `False` | If True, add a second per-joint residual on top of the per-view affine. |
| `v53_cnc_max_shift_px` | float | `32.0` | Clamp correction in pixel space. |
| `v53_cnc_loss_weight` | float | `0.01` | Weight of `cnc_loss` in total training loss. |
| `v53_cnc_warmup_epochs` | int | `0` | Freeze correction (gate = 0) for this many epochs. |
| `v53_cnc_identity_init` | bool | `True` | Zero-init final layer and gate. Must stay `True`. |

## 4. Expected MPJPE impact

- **Conservative:** −1.0 to −2.0 mm on WebBridge/H36M val sets by removing coherent calibration drift.
- **Optimistic:** −2.5 to −4.0 mm on 3DPW actual-mode, where calibration mismatch is larger.
- The gain compounds with v52 because v53 supplies cleaner rays, so v52’s precision weights become more reliable.

## 5. Risks and mitigations

See `docs/swarm_iter27/reports/agent_camera_noise_correction_v53_risks.md` for the full risk register. In brief:

- **Overfitting to camera-specific biases:** regularize with the clamp and the reprojection-change term.
- **Unstable ray correction near the principal point:** normalize by focal length and clip shifts.
- **Interaction with v52 weights:** keep v53 identity-at-init and use `v53_cnc_warmup_epochs`.

## 6. 5-step implementation plan

1. **Module file.** Create `motionflow_mv/fusion/camera_noise_correction_v53.py` implementing `CameraNoiseCorrectionV53` with the per-view affine head, gating, and auxiliary loss above.
2. **Wiring.** In `OmniMultiViewFusionV5.__init__` and `forward`, instantiate v53 when `use_v53_camera_noise_correction` is True and pass its corrected `points_2d` into the v52 UWT block.
3. **Smoke config.** Add `configs/benchmark_v53_cnc_smoke.yaml` and `scripts/run_v53_cnc_smoke_local_4090.sh` mirroring v52 conventions.
4. **Identity-at-init test.** Load a v52 checkpoint with v53 enabled and confirm `val_MPJPE` changes by < 0.1 mm.
5. **Smoke + queue.** Run the smoke test, compare to the v52 baseline, and add the full A800 entry to `scripts/launch_v33_a800_queue.py` under `v53_camera_noise_correction_on_v52`.

# v52 Test-Time Self-Refinement (TTSR)

## Motivation

The MotionFlow pipeline currently turns calibrated multi-view 2D keypoints into a 3D pose via triangulation, Gauss–Newton refinement, and a residual MLP.  Subsequent modules (v28/v40 physical loss, v46 sparse-view reliability, v50 Self-Evolution Feedback Head, v51 CDSVR) improve robustness, but the **final 3D pose is still produced by a single feed-forward pass**.  The paper narrative is

> multi-view video → human pose extraction → multi-view fusion/calibration → physical-space alignment → optimized MotionFlow,

which suggests a final *optimization* stage that closes the loop between the fused pose and the multi-view evidence.  v52 adds a lightweight, **learned test-time self-refinement** module: a small network that takes the feed-forward pose as an initial guess and refines it using reprojection residuals, bone-length cues, and temporal smoothness.  The module is **identity-at-init**, so enabling it leaves any pretrained baseline unchanged until it learns a non-zero correction.

---

## Architecture

The module is inserted **after** the final residual MLP and **before** the model returns `pred_3d`.  It can also optionally consume v50/v51 reliability/uncertainty as seed features.

### 1. Inputs

From `OmniMultiViewFusionV5`:

* `P0 ∈ R^(B,T,J,3)` — base 3D pose (e.g. output of the residual MLP).
* `x_2d ∈ R^(B,T,V,J,2)` — calibrated 2D keypoints.
* `K ∈ R^(B,T,V,3,3)`, `R ∈ R^(B,T,V,3,3)`, `t ∈ R^(B,T,V,3)` — camera intrinsics/extrinsics.
* `view_mask ∈ {0,1}^(B,T,V)` — active views.
* `r ∈ [0,1]^(B,T,V)` (optional) — per-view reliability from v46/v50/v51.
* `log σ ∈ R^(B,T,J)` (optional) — per-joint log-uncertainty from v50/v51.

### 2. Per-joint feature tokens

For each joint `j` at time `t`, we build a feature vector from four sources:

1. **Reprojection residual** (the core self-refinement signal):
   ```text
   π_v(P0_t) = K_v · [R_v · P0_t,j + t_v]_{xy} / [R_v · P0_t,j + t_v]_z
   e_vt,j = || π_v(P0_t,j) - x_2d_t,v,j ||_2^2
   e_t,j = Σ_v mask_v · r_v · e_vt,j / (Σ_v mask_v · r_v + ε)
   ```
   `e ∈ R^(B,T,J)` is robust to missing views because it is weighted by the optional reliability `r`.

2. **Bone-direction feature**:
   ```text
   b_t,j = P0_t,j - P0_t,parent(j)   (root padded with zero)
   l_t,j = ||b_t,j|| - μ_bone(j)
   ```
   giving bone-length residual `l ∈ R^(B,T,J)` and bone direction `b ∈ R^(B,T,J,3)`.

3. **Temporal smoothness feature**:
   ```text
   a_t,j = P0_t+1,j - 2 P0_t,j + P0_t-1,j   (zero-padded at boundaries)
   ```
   yielding acceleration magnitude `||a|| ∈ R^(B,T,J)`.

4. **Uncertainty seed** (optional): `u_t,j = log σ_t,j ∈ R^(B,T,J)`.

The joint token is then
```text
f_j = Linear( [P0_j, e_j, b_j, l_j, ||a_j||, u_j] ) ∈ R^(B,T,J,d)
```
where `d = v52_ttsr_hidden`.

### 3. Skeleton-aware self-attention

Tokens are processed by a 2-layer transformer **over joints**, with an skeleton-adjacency attention mask that blocks attention beyond a 2-hop neighbour distance.  This keeps the module local and interpretable:

```text
f' = TransformerEncoder(f; A)   # A_ij = -∞ if hop(i,j) > 2
```

Each layer uses `v52_ttsr_num_heads` heads, dropout `v52_ttsr_dropout`, and pre-normalization.

### 4. Residual refinement head (identity-at-init)

Two zero-initialized linear heads produce:

* `ΔP ∈ R^(B,T,J,3)` — per-joint 3D correction.
* `g ∈ R^(B,T,J,1)` — per-joint gate, passed through `2·sigmoid(g) - 1` so that at initialization `g ≈ 0` and the correction is zero.

The refined pose is
```text
P* = P0 + g ⊙ ΔP
```

Because `ΔP` and `g` are zero-initialized, `P* = P0` at startup, making the module warm-startable.

### 5. Optional iterative refinement

At inference, the same network can be applied `v52_ttsr_num_iter` times with shared weights:
```text
P^(k+1) = P^(k) + g^(k) ⊙ ΔP^(k)
```
Each step remains identity-at-init because the output layers are zero-initialized.

### 6. Losses

During training, the refinement is supervised with a 3D loss weighted by `v52_ttsr_loss_weight`:
```text
L_v52 = λ · MPJPE(P*, P_gt)
```
Optionally, a small self-supervised reprojection/bone-length loss can be added for unlabelled or mixed-domain data:
```text
L_aux = λ_reproj · L_reproj(P*; x_2d) + λ_bone · L_bone(P*) + λ_temp · L_temporal(P*)
```

---

## Integration into `OmniMultiViewFusionV5`

The call is placed immediately after the residual MLP (and after any optional v50/v51 heads that feed reliability/uncertainty):

```python
if (self.use_v52_test_time_self_refinement
        and self.test_time_self_refinement_v52 is not None):
    pred_3d = self.test_time_self_refinement_v52(
        pred_3d.view(B, T, J, 3),
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        view_mask=view_mask_flat.view(B, T, V),
        reliability=v51_reliability,      # optional
        log_uncertainty=v51_log_sigma,  # optional
    ).view(B * T, J, 3)
```

---

## Config flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v52_test_time_self_refinement` | bool | `False` | Enable the module. |
| `v52_ttsr_hidden` | int | `64` | Hidden dimension of joint tokens. |
| `v52_ttsr_num_layers` | int | `2` | Number of skeleton-aware transformer layers. |
| `v52_ttsr_num_heads` | int | `4` | Number of attention heads. |
| `v52_ttsr_dropout` | float | `0.1` | Dropout in transformer. |
| `v52_ttsr_num_iter` | int | `1` | Inference refinement iterations. |
| `v52_ttsr_hop_distance` | int | `2` | Skeleton-hop limit for attention mask. |
| `v52_ttsr_use_reliability` | bool | `True` | Use v46/v50/v51 reliability in reprojection feature. |
| `v52_ttsr_use_uncertainty` | bool | `True` | Use v50/v51 log-uncertainty as a feature. |
| `v52_ttsr_loss_weight` | float | `1.0` | Weight of `L_v52`. |
| `v52_ttsr_reproj_weight` | float | `0.0` | Weight of optional self-supervised reprojection loss. |
| `v52_ttsr_bone_weight` | float | `0.0` | Weight of optional bone-length loss. |
| `v52_ttsr_temporal_weight` | float | `0.0` | Weight of optional temporal-smoothness loss. |
| `v52_ttsr_identity_init` | bool | `True` | Zero-initialize correction and gate heads. |

---

## Expected MPJPE impact

* **WebBridge / H36M**: a 2–4 mm reduction at typical baselines (~26–30 mm), mainly on frames with large reprojection residuals.
* **MPI-INF-3DHP / 3DPW**: a 3–6 mm reduction, because the bone/temporal priors regularize domain-shifted poses.
* **Sparse-view v46 / v51**: a 1–3 mm gain by redistributing residual error across the skeleton when only 2–3 views are available.

---

## 5-step implementation plan

1. **Module stub**: create `motionflow_mv/fusion/test_time_self_refinement_v52.py` with `TestTimeSelfRefinementV52`, skeleton-aware attention, and zero-initialized output heads.
2. **Wiring in `OmniMultiViewFusionV5`**: add the config flags, instantiate the module, and insert the forward call after the residual MLP.
3. **Loss hook**: add the supervised `L_v52` (and optional self-supervised terms) to the trainer’s auxiliary loss dictionary.
4. **Smoke test**: create `configs/benchmark_v52_ttsr_smoke.yaml`, run a 50-sample smoke on the local RTX 4090, and verify that val_MPJPE does not regress when `v52_ttsr_loss_weight = 0`.
5. **Full A800 run**: add an entry to `scripts/launch_v33_a800_queue.py` and compare epoch-1 val_MPJPE against the v51 CDSVR baseline; ablate `num_iter`, `use_reliability`, and `use_uncertainty`.

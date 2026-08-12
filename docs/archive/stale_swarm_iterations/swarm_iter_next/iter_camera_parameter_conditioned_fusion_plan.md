# Iteration Plan: Camera-Parameter-Conditioned Fusion

> Prepared for the `multiview-residual-exploration` branch.  
> This is a **fallback / next-step candidate** in case the in-progress Tier-2 Bayesian Triangulation run does not beat the current 8.75 mm MPJPE anchor.

---

## 1. Motivation: why this direction can beat the 8.75 mm anchor

The current anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) reaches **8.75 mm MPJPE / 4.95 mm PA-MPJPE** on MPI-INF-3DHP S2/Seq1 (`docs/results_iter15.md`).  It already has:

- Spatio-temporal (time + view) self-attention.
- A learned principal-point / intrinsic correction head (`PrincipalPointCorrection` in `motionflow_mv/fusion/principal_point_correction.py`).
- A residual refinement MLP after weighted DLT triangulation.

What it still lacks is **explicit, learned use of the calibrated camera rig geometry** inside the two places that directly decide the final 3D pose:

1. **The view-selection weight head.** The anchor predicts `w_{v,j}` from per-view appearance features only. It cannot tell whether a view is geometrically inconsistent because of calibration drift, occlusion, or bad 2D detection.
2. **The residual refinement head.** The anchor residual MLP receives only the pooled feature `f` and the raw triangulation `X_raw`. It does not know the rig baseline, focal length, or camera ray angles, so its correction cannot be normalized to the physical scale of the scene.

Injecting `K, R, t` into these two heads should:

- Down-weight views whose rays are geometrically inconsistent with the rest of the rig (e.g., after camera perturbation).
- Make the residual correction scale-aware, reducing systematic errors on joints such as hips and hands, which the failure analysis flagged as the worst (`docs/swarm_iter_next/failure_analysis_crossview_pp.md`).
- Improve cross-dataset transfer, because camera parameters replace dataset-specific learned view embeddings.

Empirical precedent in this repo already points in this direction:

- Adding camera-aware ray embeddings + principal-point correction dropped the baseline from 25.2 mm to **9.32 mm** (`docs/paper_draft_icra_cvpr_2027.md`, Table 5.1).
- Direct principal-point supervision during a robust re-train further improved it to **8.75 mm**.
- `CameraPositionalEncoding` (`motionflow_mv/fusion/camera_positional_encoding.py`) showed that geometry-based camera tokens can replace learned view embeddings while keeping accuracy within 1 mm of the learned-embedding baseline (`docs/swarm_iter_next/design_camera_positional_encoding_plan.md`).

The next logical step is therefore to **condition the *decision* heads on the camera parameters**, not just the *feature* extractor.  This is a small, low-risk architectural change (~1–2 k extra parameters), but it attacks exactly the residual error that the 8.75 mm anchor still leaves on the table.

---

## 2. Architecture

### 2.1 High-level model

Start from the anchor class

```text
motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
```

and replace two submodules:

1. `self.weight_head` → `_CameraConditionedWeightHead`
2. `self.residual_mlp` → `_CameraConditionedResidualRefiner`

Everything else (principal-point correction, spatio-temporal transformer, DLT triangulation, output tuple) stays identical.  An implementation already exists at:

```text
motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_model.py
    RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned
```

### 2.2 Camera-conditioned view-weight head

Input:

- `feat ∈ ℝ^(N,J,V,d)` — per-joint per-view features after spatio-temporal attention.
- `K, R, t` — camera intrinsics/extrinsics, shapes `(N,V,3,3)`, `(N,V,3,3)`, `(N,V,3)`.

Forward:

```
cam_feat_v = concat([ flatten(K_v), flatten(R_v), flatten(t_v) ])   # ℝ^(V,21)
cond_v     = MLP_cam(cam_feat_v)                                    # ℝ^(V, c)
cond_vj    = broadcast(cond_v) over J                                # ℝ^(J,V,c)
x_vj       = concat([feat_vj, cond_vj])                             # ℝ^(J,V,d+c)
logit_vj   = MLP_weight(x_vj) ∈ ℝ^(J,V)
```

Code reference: `_CameraConditionedWeightHead.forward` in `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_model.py:51-74`.

### 2.3 Camera-conditioned residual refiner

Input:

- `feat_pool ∈ ℝ^(N,J,d)` — pooled spatio-temporal feature.
- `X_raw ∈ ℝ^(N,J,3)` — raw weighted-DLT triangulation.
- `K, R, t` as above.

Forward:

```
cam_feat  = concat([ flatten(K_v), flatten(R_v), flatten(t_v) ])   # ℝ^(V,21)
cond      = MLP_rig( mean_v cam_feat )                             # ℝ^(c_rig)
cond_j    = broadcast(cond) over J                                 # ℝ^(J, c_rig)
ΔX        = MLP_delta( concat([feat_pool_j, X_raw_j, cond_j]) )    # ℝ^(J,3)
X         = X_raw + ΔX
```

Code reference: `_CameraConditionedResidualRefiner.forward` in `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_model.py:106-129`.

### 2.4 Optional auxiliary losses

The repo already contains drop-in camera-conditioned auxiliaries in `motionflow_mv/losses/camera_conditioned_loss.py`:

- `camera_conditioned_reprojection_loss(pred_3d, points_2d, K, R, t, weights, mask, loss_type="charbonnier")`
- `camera_conditioned_scale_loss(pred_3d, parents)`

These can be added to the training objective without touching existing loss code.

### 2.5 Optional upgrade: geometry-based camera positional encoding

The learned `view_pos_embed` in the anchor is dataset-specific.  As an optional second step, replace it with `CameraPositionalEncoding` (`motionflow_mv/fusion/camera_positional_encoding.py`):

```python
view_emb = self.camera_pos_enc(K, R, t)  # (N*V, d)
feat = feat + view_emb[:, None, :, None, :]
```

This makes the model variable-view and cross-dataset transferable, following the plan in `docs/swarm_iter_next/design_camera_positional_encoding_plan.md`.  It is **not** required for the first MPI-INF-3DHP full run, but it is the natural follow-up if the clean result matches the anchor and transfer is the target.

---

## 3. Code changes needed

### 3.1 Create/extend the model file (already partly implemented)

- **Existing:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_model.py`
  - Contains `_CameraConditionedWeightHead`, `_CameraConditionedResidualRefiner`, and `RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned`.
  - The `forward` pass injects `K, R, t` into both heads at lines 264 and 276.

- **Recommended tweak for the next full run:** add a small learnable **ray-angle / baseline embedding** inside `_CameraConditionedWeightHead` so the weight head can reason about epipolar compatibility, not just raw matrix entries.  Concretely, encode the angle between the principal ray of each view and the mean camera-center-to-joint ray:

```python
# Inside _CameraConditionedWeightHead.forward
joints_norm = ...  # (N, J, V, 3) camera-center-to-joint direction
ray_angle = torch.einsum("nvj,nvj->nv", principal_rays, joints_norm)  # (N,V)
ray_angle = ray_angle.unsqueeze(1).expand(N, J, V)                   # (N,J,V)
```

  Append `ray_angle[..., None]` to `cam_feat` before the encoder.  This is optional but cheap and directly targets the geometric failure mode.

### 3.2 Add a FusionModule wrapper and register it

Create:

```text
motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_module.py
```

mirroring `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_module.py`:

- `RayAttentionTemporalCrossviewResidualCameraConditionedFusionModule(FusionModule)`
- `name = "ray_attention_temporal_crossview_residual_camera_conditioned"`
- `fuse(...)` takes `points_2d`, `confidences`, and `cameras` (list of `Camera`), builds the `(T,V,J,3)` input tensor, and calls the model with `cameras=cameras`.
- A `register_...()` function to add it to `FUSION_REGISTRY`.

Then add the import and registration call to:

```text
motionflow_mv/fusion/__init__.py
```

### 3.3 Wire the model into the main training script

Modify:

```text
experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py
```

- Add `"camera_conditioned"` to the `model_type` choices.
- Import `RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned` at the top.
- Add a new `elif args.model_type == "camera_conditioned":` branch that instantiates it with the same hyperparameters as the anchor.
- Add command-line flags:
  - `--camera_conditioned_reproj_weight` (default `0.0`)
  - `--camera_conditioned_scale_weight` (default `0.0`)
- In the training loop, when the model is `camera_conditioned`, optionally call:

```python
from motionflow_mv.losses.camera_conditioned_loss import (
    camera_conditioned_reprojection_loss,
    camera_conditioned_scale_loss,
)

if args.camera_conditioned_reproj_weight > 0.0:
    loss = loss + args.camera_conditioned_reproj_weight * camera_conditioned_reprojection_loss(
        pred, xb[..., :2], K, R, t, weights=None
    )

if args.camera_conditioned_scale_weight > 0.0:
    loss = loss + args.camera_conditioned_scale_weight * camera_conditioned_scale_loss(
        pred, parents=MPI_INF_3DHP_28_PARENTS  # or H36M_17_PARENTS as appropriate
    )
```

The existing `velocity_loss`, `pp_loss_weight`, and camera-augmentation code paths should remain untouched.

### 3.4 Add to the evaluation harness

Modify:

```text
experiments/eval_full_metrics.py
```

- Import `RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned`.
- Add to `MODEL_CLASSES`:

```python
"camera_conditioned_pp": RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned,
```

- The `build_model` function already handles `crossview_residual_pp`-like models; add `camera_conditioned_pp` to the same branch that sets `n_st_layers` and `residual_hidden`.

### 3.5 Add a smoke test

Create or extend:

```text
tests/test_camera_parameter_conditioned_fusion.py
```

Tests should cover:

- Forward/backward shape consistency for `(B,T,V,J,3)` input.
- Camera argument handling: passing `cameras=` list vs. `K=..., R=..., t=...` tensors.
- That the model returns the same output tuple as the anchor (`pred_3d`, `weights`, optionally `pp_delta`, `focal_scale`, `raw_3d`).
- That training one step with the camera-conditioned reprojection loss produces finite gradients.

Reuse the helper pattern in `tests/test_iter14_models_train_step.py`.

---

## 4. Training & evaluation protocol

### 4.1 Datasets

Use the same canonical `.npz` format as the anchor and the Bayesian run:

- **Train:**
  - `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz`
- **Validation:**
  - `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`
- **Cross-dataset sanity (optional, after full MPI run):**
  - `data/h36m_hf/s_01_acts_*_multiview.npz`

### 4.2 Loss

Primary objective:

```
L = L_3D_MSE
    + λ_pp * L_pp_correction
    + λ_reproj * L_camera_conditioned_reprojection
    + λ_scale * L_camera_conditioned_scale
    + λ_vel * L_velocity
```

Recommended starting hyperparameters (mirror the anchor and Bayesian run):

| Hyperparameter | Value | Notes |
|---|---|---|
| `clip_len` | 13 | Same as anchor. |
| `d` | 64 | Same as anchor. |
| `residual_hidden` | 128 | Same as anchor. |
| `n_st_layers` | 2 | Same as anchor. |
| `camera_condition_dim` | 32 | Default in model file. |
| `residual_condition_dim` | 32 | Default in model file. |
| `pp_loss_weight` | 0.2 | Same as Bayesian full run. |
| `cam_aug_pp` | 5.0 px | Same as Bayesian full run. |
| `cam_aug_focal` | 0.01 | Same as Bayesian full run. |
| `cam_aug_schedule` | `intrinsics_curriculum` | Ramp PP/focal over first 5 epochs. |
| `pp_pretrain_epochs` | 3 | Stabilizes the PP head before full end-to-end training. |
| `camera_conditioned_reproj_weight` | 0.0 initially; try 0.05 if smoke is stable. |
| `camera_conditioned_scale_weight` | 0.0 initially; try 0.01 if smoke is stable. |

### 4.3 Metrics

Report the standard metrics computed by `motionflow_mv/eval/metrics.py`:

- `mpjpe` (mm)
- `pa_mpjpe` (mm)
- `pck@50mm`, `pck@100mm`, `pck@150mm`
- `pck_auc` (0–150 mm)
- `per_joint_mpjpe`

Use the `BenchmarkProtocol` in `motionflow_mv/eval/benchmark_protocol.py` for a reproducible multi-seed evaluation if time permits.

### 4.4 Robustness matrix

Because the main value of camera conditioning is robustness to calibration drift, run the same perturbation grid used for the anchor (`docs/paper_draft_icra_cvpr_2027.md`, Section 5.1):

| Perturbation | Magnitude | Why it matters |
|---|---|---|
| Clean | — | Anchor: 8.75 mm. |
| Rotation | ±0.5°, ±1.0° | Largest failure mode of the anchor. |
| Translation | ±5 mm, ±10 mm | Tests rig-scale awareness. |
| Focal length | ±1%, ±2% | Tests intrinsic-aware weighting. |
| Principal point | ±3 px, ±5 px | Tests the PP correction synergy. |
| View dropout | 2 views, 3 views | Tests geometry-aware view selection. |

The `evaluate()` function in `experiments/eval_full_metrics.py` can be reused; for variable views, wrap the model with `VariableViewInferenceWrapper` (`motionflow_mv/fusion/variable_view_inference.py`).

### 4.5 Baseline to compare

- **Anchor checkpoint:** `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth`
- **Anchor metrics:** 8.75 mm MPJPE, 4.95 mm PA-MPJPE on MPI-INF-3DHP S2/Seq1.
- **In-progress competitor:** `outputs/bayesian_tri_pp_full_mpiinf3dhp.pth` (Tier-2 Bayesian triangulation).  Only start the full camera-conditioned run if Bayesian does not beat 8.75 mm.

---

## 5. Expected gains and risks

### 5.1 Expected gains

- **Clean MPI-INF-3DHP:** modest improvement, estimated **8.3–8.6 mm MPJPE** (0.1–0.4 mm over the anchor).  The gain comes from geometry-aware residual correction on hips/hands.
- **Calibration robustness:** larger relative improvement.  Camera-conditioned weighting should reduce the degradation under rotation/translation perturbations by **10–20%** compared with the anchor.
- **Variable-view inference:** if the optional camera positional encoding is enabled, the model should degrade more gracefully when dropping from 14 to 2–4 views.
- **Cross-dataset transfer:** replacing dataset-specific `view_pos_embed` with geometry-based tokens should let the same checkpoint run on H36M (4 views) and MPI-INF-3DHP (14 views) without shape errors.

### 5.2 Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Overfitting the small MPI train set with extra camera-conditioned parameters. | Medium | Keep `camera_condition_dim` small (32); use the same camera augmentation as the anchor; monitor val MPJPE and stop early. |
| Harder initialization of the weight head delays convergence. | Medium | Warm-start the backbone from the 8.75 mm anchor checkpoint (`strict=False`), then train only the new heads for 1–2 epochs before unfreezing. |
| Camera augmentation too strong drowns the camera-conditioned signal. | Low | Reuse the `intrinsics_curriculum` schedule proven by the Bayesian run. |
| Gains are limited because the anchor already uses ray embeddings. | Medium | If smoke does not improve after 5 epochs, fall back to the anchor and document camera conditioning as a negative ablation. |
| GPU is occupied by the Bayesian run. | Certain | Do not start full training until the Bayesian run finishes or is deprioritized. The smoke test can run on CPU or on a second GPU if available. |

---

## 6. Next steps for the follow-up implementer

1. **Integrate the model into the pipeline** (≈ 30 min):
   - Add `RayAttentionTemporalCrossviewResidualCameraConditionedFusionModule` and register it in `motionflow_mv/fusion/__init__.py`.
   - Add the `"camera_conditioned"` branch and loss flags in `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`.
   - Register the model in `experiments/eval_full_metrics.py`.

2. **Run a fast CPU/GPU smoke test** ( 1–2 h on RTX 4090):

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --model_type camera_conditioned \
  --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --epochs 5 --batch_size 8 --lr 1e-3 \
  --pp_loss_weight 0.2 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule intrinsics_curriculum \
  --pp_pretrain_epochs 3 \
  --output outputs/camera_conditioned_pp_smoke.pth
```

   Compare the final val MPJPE to the anchor's 5-epoch smoke (~9.1–9.3 mm).  If it is not better, try warm-starting from the anchor checkpoint for 2–3 epochs.

3. **Launch the full 20-epoch run** only after the Bayesian Triangulation run is judged unsuccessful and GPU is free.  Mirror the Bayesian command but with `model_type=camera_conditioned` and no `epipolar_loss_weight`.

4. **Evaluate the full checkpoint** with `experiments/eval_full_metrics.py --model camera_conditioned_pp` and the robustness matrix.  Update `docs/results_iter15.md` (or the next iter results file) with the new numbers.

5. **Optional extension:** if the clean result is promising but variable-view/cross-dataset transfer is poor, enable `CameraPositionalEncoding` as described in Section 2.5 and re-run the smoke test.

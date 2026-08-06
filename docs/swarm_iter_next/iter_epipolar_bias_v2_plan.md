# Iteration Plan: Epipolar Bias v2

**Direction:** `epipolar_bias_v2`  
**Goal:** Beat the current 8.75 mm MPJPE anchor on MPI-INF-3DHP by injecting calibrated epipolar geometry as a *learned relative-position bias* inside the spatio-temporal transformer.  
**Anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` (8.75 mm MPJPE on MPI-INF-3DHP S2/Seq1).  
**Status:** The model class already exists in `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model.py`, but it is **not yet wired** into the trainer or evaluation harness.

---

## 1. Motivation

The anchor model (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) fuses multi-view 2-D keypoints with a joint spatio-temporal transformer over `(time, view)` tokens and then triangulates.  It learns everything from data, so it has no explicit geometric inductive bias that tells it two views are inconsistent when their keypoints violate the epipolar constraint.

Two related baselines already exist:

* **v1 epipolar weight-head bias** (`ray_attention_temporal_crossview_residual_principal_point_epipolar_model.py`) computes the symmetric epipolar-line distance after feature fusion and biases the *final per-view weight logits*.  This helps triangulation but leaves the transformer itself geometry-agnostic.
* **Bayesian triangulation** (`ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`) predicts anisotropic 2-D covariances and runs adaptive Gauss-Newton refinement.  It is the current Tier-2 candidate running on the RTX 4090.

**Why v2 can beat the anchor:**

1. **Geometry-aware feature fusion.**  By adding an epipolar relative-position bias directly to the attention scores of every ST transformer layer, views that are geometrically consistent attend to each other more strongly *before* the weight head or triangulation see them.  This is a stronger prior than biasing only the final weights.
2. **Built-in robustness to noisy/occluded views.**  When a view is noisy, the epipolar distance to the other views grows; the bias suppresses that view’s tokens in the transformer, reducing its influence on the pooled features used by the residual MLP.
3. **Minimal, ablatable change.**  The v2 model subclasses the anchor and reuses the same triangulation, weight head, and residual MLP.  A single learned scalar gate (`epipolar_gate`) can suppress the bias entirely, so the network can fall back to the anchor if the geometry term hurts.
4. **Composability.**  The v2 bias is orthogonal to the Bayesian triangulation covariance/GN head; if the running Bayesian run does not beat 8.75 mm, v2 can be stacked on top of it as a follow-up.

---

## 2. Architecture

### 2.1 High-level data flow

```
Input x: (B, T, V, J, 3)   [u, v, confidence]
        |
        v
PrincipalPointCorrection
        |
        v
_extract_frame_features(...) -> feat: (B*T, V, J, d)
        |
        v
+ time/view positional embeddings
        |
        v
EpipolarBiasedTransformerEncoderLayer x n_st_layers
   (with per-frame epipolar attention bias)
        |
        v
weight_head + sigmoid -> weights: (B*T, V, J)
        |
        v
_triangulate_weighted_dlt -> pred_3d_raw
        |
        v
residual_mlp(delta) -> pred_3d = pred_3d_raw + delta
```

### 2.2 Key modules and equations

**Module:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model.py`

```python
class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    def __init__(..., epipolar_temperature=10.0, gate_init=2.0):
        ...
        self.epipolar_gate = nn.Parameter(torch.full((1,), gate_init))
        self.st_transformer = nn.ModuleList([
            EpipolarBiasedTransformerEncoderLayer(...)
            for _ in range(n_st_layers)
        ])
```

**Geometry helpers:** `motionflow_mv/fusion/epipolar_transformer_bias.py`

* `_aggregate_pairwise_epipolar_distance(K, R, t, points_2d)`  
  Computes symmetric epipolar-line distance for every view pair and every joint:

  ```
  d_epi[v_src, v_dst, j] = 0.5 * (dist[v_src, v_dst, j] + dist[v_dst, v_src, j])
  ```

* `compute_per_frame_epipolar_bias(K, R, t, points_2d, temperature=10.0)`  
  Averages over joints to obtain a per-view-pair bias:

  ```
  bias[n, v_src, v_dst] = - mean_j d_epi[n, v_src, v_dst, j] / temperature
  ```

* `build_temporal_bias_from_frames(per_frame_bias, n_heads, n_joints)`  
  Expands the per-frame `(V, V)` bias into the ST-attention mask shape:

  ```
  attn_bias: (B * J * n_heads, T*V, T*V)
  ```

  The mask is block-diagonal in time: same-frame view pairs get the epipolar bias; cross-frame pairs get zero bias.

**Attention score update inside `EpipolarBiasedTransformerEncoderLayer.forward`:**

```python
scores = Q @ K^T / sqrt(d_k) + gate * attn_bias
gate = sigmoid(self.epipolar_gate)
```

Because `attn_bias` is negative when views are inconsistent, inconsistent view tokens receive lower attention weight.

### 2.3 Design knobs

| Symbol | Default | Meaning |
|---|---|---|
| `epipolar_temperature` | 10.0 | Smaller → sharper geometry bias (px). |
| `gate_init` | 2.0 | Initial logit; `sigmoid(2.0) ≈ 0.88` keeps the bias active from the start. |
| `n_st_layers` | 2 | Number of epipolar-biased ST layers (matches anchor). |

---

## 3. Code changes needed

All changes are additive.  Do **not** modify the running Bayesian triangulation experiment.

### 3.1 Trainer integration

**File:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`

1. Import the v2 model class:

   ```python
   from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model import (
       RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2,
   )
   ```

2. Add `"epipolar_bias_v2"` to the `--model_type` `choices` list (around the existing `choices=["temporal", ..., "bayesian_tri"]`).

3. Add a new branch in `main()` after the existing `epipolar` branch:

   ```python
   elif args.model_type == "epipolar_bias_v2":
       model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2(
           j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
           residual_hidden=args.residual_hidden,
           principal_point_hidden=args.principal_point_hidden,
           principal_point_max_offset=args.principal_point_max_offset,
           focal_max_scale=args.focal_max_scale,
           return_pp_delta=True,
       ).to(device)
   ```

### 3.2 Evaluation harness integration

**File:** `experiments/eval_full_metrics.py`

1. Import the v2 model:

   ```python
   from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model import (
       RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2,
   )
   ```

2. Add to `MODEL_CLASSES`:

   ```python
   "epipolar_bias_v2_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2,
   ```

3. Extend `build_model()` so that `epipolar_bias_v2_pp` receives `n_st_layers` and `residual_hidden`, and add `principal_point_hidden` / `principal_point_max_offset` kwargs.

### 3.3 (Optional) Fusion-module wrapper

For full pipeline integration via `FusionModule` / `FUSION_REGISTRY`:

* Create `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_module.py` mirroring `ray_attention_temporal_crossview_residual_principal_point_module.py` but wrapping the v2 model class.
* Register it in `motionflow_mv/fusion/__init__.py`.

This is only needed if the v2 checkpoint is to be consumed by the generic inference server; training/eval can proceed without it.

### 3.4 Convenience scripts

Add under `scripts/` (do not run the full script while Bayesian training is in progress):

* `scripts/run_epipolar_bias_v2_pp_smoke_wsl.sh` — 5-epoch smoke test.
* `scripts/run_epipolar_bias_v2_pp_full_wsl.sh` — 10–20 epoch full run.
* `scripts/eval_epipolar_bias_v2_pp_full_wsl.sh` — full-metrics evaluation using `experiments/eval_full_metrics.py --model epipolar_bias_v2_pp`.

### 3.5 Configuration file

Add `configs/train_epipolar_bias_v2_full.yaml` with the hyper-parameter set below (the training scripts currently use argparse, but a YAML snapshot is useful for reproducibility and the project already stores configs in `configs/`).

---

## 4. Training & evaluation protocol

### 4.1 Data

Use the canonical WebBridge `.npz` files produced by `motionflow_mv/data/webbridge_loader.py`:

* **Train:**
  * `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz`
  * `data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz`
  * `data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz`
  * `data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz`
* **Val:**
  * `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`

Loader: `TemporalClipDataset` / `RandomClipDataset` in `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`.

### 4.2 Hyper-parameters

| Setting | Value |
|---|---|
| `clip_len` | 13 |
| `d` | 64 |
| `n_st_layers` | 2 |
| `residual_hidden` | 128 |
| `principal_point_hidden` | 64 |
| `principal_point_max_offset` | 20.0 px |
| `batch_size` | 8 |
| `lr` | 1e-3 |
| `pp_loss_weight` | 0.1 |
| `cam_aug_schedule` | `intrinsics_curriculum` |
| `cam_aug_pp` | 5.0 px |
| `cam_aug_focal` | 0.01 |
| `epipolar_temperature` | 10.0 (default; sweep 5–20 if smoke shows sensitivity) |
| `gate_init` | 2.0 |

### 4.3 Smoke test

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --batch_size 4 --train_samples 500 --epochs 5 \
  --model_type epipolar_bias_v2 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 \
  --output outputs/epipolar_bias_v2_pp_smoke.pth
```

Expected runtime: ~50–70 min on the local RTX 4090 (similar to the v1 epipolar smoke).

### 4.4 Full training

10 epochs, 1000 random clips per train sequence:

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --epochs 10 --train_samples 1000 --batch_size 8 --val_stride 50 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
  --model_type epipolar_bias_v2 \
  --output outputs/ray_attention_temporal_crossview_residual_pp_epipolar_bias_v2_full.pth
```

### 4.5 Evaluation

Clean metrics:

```bash
python experiments/eval_full_metrics.py \
  --model epipolar_bias_v2_pp \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --checkpoint outputs/ray_attention_temporal_crossview_residual_pp_epipolar_bias_v2_full.pth \
  --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
  --val_stride 50 \
  --output_json outputs/epipolar_bias_v2_pp_full_eval.json
```

Metrics computed by `motionflow_mv/eval/metrics.py`: MPJPE, PA-MPJPE, root-relative MPJPE, velocity MPJPE, PCK@50/100/150, PCK-AUC, per-joint MPJPE.

Robustness metrics (re-use the existing occlusion/robustness script):

* Add synthetic noise `noise_std=2.0`, outliers `outlier_rate=0.05`, view dropout `0.25`.
* Compare against the anchor and v1 epipolar checkpoints under identical perturbations.

### 4.6 Baselines to compare

| Model | File | Why |
|---|---|---|
| PP anchor | `ray_attention_temporal_crossview_residual_principal_point_model.py` | 8.75 mm MPJPE anchor. |
| v1 epipolar weight-head | `ray_attention_temporal_crossview_residual_principal_point_epipolar_model.py` | Geometry only at the weight head. |
| Bayesian triangulation | `ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py` | Current Tier-2 run; may be combined later. |

---

## 5. Expected gains and risks

### 5.1 Expected gains

* **Clean MPI-INF-3DHP S2/Seq1:** 0.15–0.40 mm MPJPE improvement over the 8.75 mm anchor, landing in the 8.45–8.60 mm range.  The prior v1 epipolar run reported ~9.28 mm vs. the 9.32 mm anchor; moving the bias from the weight head into the transformer should provide an additional modest gain.
* **Noisy/perturbed data:** Larger relative improvement (3–6 % lower MPJPE) because the epipolar bias explicitly down-weights inconsistent views before triangulation.
* **Diagnostic confirmation:** `sigmoid(epipolar_gate)` should converge to > 0.5, confirming the model uses the geometry term.

### 5.2 Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Shape mismatch** in `attn_bias` (currently `(B*J*n_heads, T*V, T*V)`) | The helper `build_temporal_bias_from_frames` already matches `MultiheadAttention` expectations; run a CPU forward pass before any GPU training. |
| **Over-regularization:** a strong negative bias may suppress useful but noisy views. | `epipolar_temperature` default 10.0 is conservative; the learned gate can also suppress the bias. |
| **Gate collapse:** the gate could learn to near-zero, reducing v2 to the anchor. | Initialize with `gate_init=2.0`; monitor `sigmoid(epipolar_gate)` in logs. |
| **Calibration errors:** epipolar bias assumes the (possibly corrected) intrinsics/extrinsics are accurate. | Pair with the existing `PrincipalPointCorrection` head and intrinsics curriculum that is already used by the anchor. |
| **GPU contention:** a full Bayesian triangulation run is already in progress. | Do not start the v2 full run until the GPU is free; the smoke test can be done on CPU or queued after the current job. |

### 5.3 Fallback plan

If the ST-attention bias does not improve over the anchor:

1. Fall back to the v1 epipolar weight-head bias (one-line model swap).
2. Combine the v2 bias with the v1 weight-head bias in a single model.
3. Combine v2 with the Bayesian covariance/Gauss-Newton head once that run finishes.

---

## 6. Next steps

1. **Wire the trainer** (`experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`): add the import, argparse choice, and model instantiation branch.
2. **Wire the evaluator** (`experiments/eval_full_metrics.py`): add `epipolar_bias_v2_pp` to `MODEL_CLASSES` and `build_model()`.
3. **CPU sanity check** (no GPU needed): instantiate the v2 model and run a single forward/backward pass with a 2-frame synthetic clip to verify `attn_bias` shape and gradient flow.
4. **Smoke run** (≤ 1 GPU hour, only after the Bayesian run finishes or on a different GPU): execute the 5-epoch smoke command and compare val MPJPE to the anchor smoke under identical settings.
5. **Full training** (10 epochs on RTX 4090, ~6–8 hours): run the full command, save checkpoint, and run `eval_full_metrics.py`.
6. **Robustness ablation**: run the same checkpoint through the existing occlusion/noise protocol and compare to the anchor and v1 epipolar.
7. **If successful**, optionally add the `FusionModule` wrapper and register it in `motionflow_mv/fusion/__init__.py` so the checkpoint can be loaded by the generic inference pipeline.

---

## 7. References

* `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model.py`
* `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_model.py`
* `motionflow_mv/fusion/epipolar_transformer_bias.py`
* `motionflow_mv/fusion/epipolar_attention_bias.py`
* `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
* `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
* `experiments/eval_full_metrics.py`
* `motionflow_mv/eval/metrics.py`
* `docs/swarm_iter15/proposal_epipolar-geometry-transformer-bias-v2.md` (prior proposal)

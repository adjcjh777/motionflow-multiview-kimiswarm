# Uncertainty-Aware Multi-View Triangulation with Anisotropic Covariance and Adaptive Gauss-Newton

**Iter15 proposal — ICRA/CVPR 2027 track**

---

## 1. One-sentence hypothesis

By predicting anisotropic 2-D image-space covariances per view/joint and using them to drive a covariance-conditioned adaptive Gauss-Newton refinement with an geometry-regularized epipolar consistency loss, the model can retain the anchor's clean accuracy while improving robustness to noisy views, calibration drift, and partial occlusions.

---

## 2. Related existing files/modules

- **Anchor model:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
- **Parent cross-view residual model:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
- **Base cross-view attention model:** `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`
- **Principal-point correction:** `motionflow_mv/fusion/principal_point_correction.py`
- **DLT triangulation:** `motionflow_mv/fusion/ray_attention_model.py` (`_triangulate_weighted_dlt`)
- **Epipolar utilities:** `motionflow_mv/fusion/epipolar_attention_bias.py`
- **Training script:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
- **Iter14 predecessor proposal:** `docs/swarm_iter14/proposal_perview-uncertainty-for-triangulation.md`

---

## 3. Proposed code changes

### 3.1 New model file

Create `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py` containing:

```python
class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
```

### 3.2 Signature changes

Add to `__init__`:

```python
def __init__(
    ...,
    covariance_hidden: int = 64,
    gn_iters: int = 2,
    min_gn_damping: float = 1e-6,
    max_gn_damping: float = 1e-2,
    epipolar_loss_weight: float = 0.05,
    return_covariance: bool = False,
):
```

### 3.3 New heads

1. **Anisotropic covariance head:**
   - Input: spatio-temporal feature `feat` of shape `(B*T, V, J, d)`.
   - Output: lower-triangular Cholesky factor `L` of shape `(B*T, V, J, 2, 2)`.
   - Implementation:
     ```python
     self.covariance_head = nn.Sequential(
         nn.Linear(d, covariance_hidden),
     nn.ReLU(),
     nn.Linear(covariance_hidden, 3),  # l_xx, l_xy, l_yy
     )
     ```
   - Constrain `l_xx > 0`, `l_yy > 0` via softplus; build positive-definite `Σ = L L^T`.

2. **Adaptive Gauss-Newton damping head:**
   - Input: pooled per-joint feature `feat_pooled` `(B*T, J, d)`.
   - Output: per-joint damping scalar `λ` in `[min_gn_damping, max_gn_damping]` via sigmoid.
   - Implementation:
     ```python
     self.damping_head = nn.Sequential(
         nn.Linear(d, covariance_hidden),
         nn.ReLU(),
         nn.Linear(covariance_hidden, 1),
         nn.Sigmoid(),
     )
     ```

### 3.4 Forward-flow changes

After the existing spatio-temporal transformer:

1. Predict `L` from per-view features.
2. Convert to scalar precision weight:
   ```python
   log_det = torch.logdet(L @ L.transpose(-2, -1))  # (B*T, V, J)
   precision = torch.exp(-0.5 * log_det)
   weights = torch.sigmoid(w_logits).permute(0, 2, 1) * confidences * precision
   ```
3. Triangulate raw 3-D pose with existing weighted DLT (`_triangulate_weighted_dlt`).
4. Refine with adaptive Gauss-Newton:
   - Predict per-joint `λ` from `feat_pooled`.
   - Minimize weighted reprojection error using the precision from the covariance head.
   - Apply predicted `λ` as diagonal damping in the normal equations.
5. Run existing residual MLP on `[feat_pooled, refined_3d]`.
6. Optionally return the covariance factors for visualization/diagnostics.

### 3.5 Auxiliary losses

Add `epipolar_consistency_loss` in the model:

```python
def _epipolar_consistency_loss(self, points_2d, K, R, t, L):
    # Compute pairwise epipolar distance for every (src, dst) view pair.
    # Weight each pair's residual by the harmonic mean of the source and
    # destination covariances (determinant).
    # Return mean loss over all view pairs and joints.
```

The training script will add:

```python
loss = criterion(pred, yb)
if epipolar_loss_weight > 0:
    loss = loss + epipolar_loss_weight * outputs[-1]
```

### 3.6 Files to create / modify

| Path | Action |
|---|---|
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py` | New model class |
| `motionflow_mv/fusion/__init__.py` | Optional: import the new model for registry convenience |
| `experiments/train_ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_mpiinf3dhp.py` | New thin training wrapper (copy of the PP trainer with new model and loss args) |
| `scripts/run_bayesian_tri_smoke_wsl.sh` | Shell one-liner for the smoke run |

---

## 4. Training/smoke plan

Run a 5-epoch smoke on MPI-INF-3DHP using the lightweight configuration consistent with previous smokes:

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
          data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --epochs 5 --train_samples 500 --batch_size 2 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 \
  --epipolar_loss_weight 0.05 --gn_iters 2
```

**Estimated runtime on RTX 4090:** ~10–15 minutes for 5 epochs (adaptive GN adds ~20% overhead, still well under 5 GB VRAM with `d=32`).

### Smoke pass/fail criteria

- **Pass:** val MPJPE finite and ≤ 13.0 mm after 5 epochs (smoke threshold; tiny config).
- **Pass:** no NaN/Inf in loss or in any returned tensor.
- **Pass:** covariance Cholesky diagonals remain positive and finite.
- **Pass:** epipolar loss is finite and non-zero.
- **Fail:** val MPJPE > 16.0 mm or any training crash/NaN.
- **Fail:** adaptive damping collapses to a constant (std across joints < 1e-4).

---

## 5. Success metrics

| Metric | Target |
|---|---|
| Clean MPJPE on MPI-INF-3DHP S2/Seq1 | ≤ 9.6 mm (within ~3% of anchor 9.32 mm) |
| PA-MPJPE | ≤ 5.8 mm |
| Robustness to view dropout (40% dropout) | ≥ 10% better than PP anchor |
| Robustness to cxcy perturbation (5 px) | ≥ 8% better than PP anchor |
| Covariance calibration | Precision-weighted triangulation outperforms uniform weights by ≥ 5% on corrupted clips |
| Inference latency on RTX 4090 | ≤ 1.5× anchor latency for a 13-frame clip |

---

## 6. Risk and fallback

| Risk | Mitigation / Fallback |
|---|---|
| Anisotropic covariance head overfits or collapses | Initialize the covariance head to produce near-identity covariance; freeze it for the first 2 epochs |
| Adaptive Gauss-Newton introduces training instability | Set `gn_iters=0` and rely only on the covariance-weighted DLT + residual MLP |
| Epipolar loss dominates clean accuracy | Reduce `epipolar_loss_weight` to 0.01 or remove it entirely |
| Clean accuracy regresses > 3% | Warm-start from the anchor checkpoint and train only the new heads for 5 epochs |
| Determinant computation is numerically unstable | Add `eps=1e-6` to the diagonal of `Σ` before `logdet` |
| Runtime exceeds target | Reduce `gn_iters` to 1 or skip adaptive GN at inference |

---

**Related anchor files:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`, `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`, `docs/swarm_iter14/proposal_perview-uncertainty-for-triangulation.md`.

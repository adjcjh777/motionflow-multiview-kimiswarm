# Per-View Uncertainty Estimation for Triangulation

**Iter14 proposal — ICRA/CVPR 2027 track**

---

## 1. Problem

The current 9.32 mm anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) predicts only a scalar per-view/joint weight for triangulation, which cannot distinguish between *low-confidence detections* and *camera/view-specific uncertainty*, causing degraded 3-D pose when a subset of views is noisy, occluded, or poorly calibrated.

## 2. Hypothesis

Adding a learnable per-view/per-joint log-variance head to the cross-view PP model, and using the predicted precision to re-weight the DLT triangulation, will keep clean accuracy within 0.3 mm of the anchor while improving robustness under view dropout, focal perturbations, and cxcy errors.

## 3. Method

### 3.1 Architecture changes

Create a new model file:

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_uncertainty_model.py`

Subclass `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` and add a small per-view/per-joint uncertainty head:

```python
self.uncertainty_head = nn.Sequential(
    nn.Linear(d, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

After the spatio-temporal transformer, for feature tensor `feat` of shape `(B*T, V, J, d)`:

1. Predict `log_var = self.uncertainty_head(feat).squeeze(-1)` → `(B*T, V, J)`.
2. Clamp `log_var ∈ [log_var_min, log_var_max]` (default `[-10, 10]`).
3. Compute per-view weights:

```python
precision = torch.exp(-log_var)
weights = torch.sigmoid(w_logits).permute(0, 2, 1) * confidences * precision
```

4. Feed `weights` into the existing weighted DLT and residual refinement.

Optionally concatenate `log_var` into the residual MLP input so the refinement head sees the uncertainty map:

```python
residual_input = torch.cat([feat_pooled, pred_3d_raw, log_var.mean(-1, keepdim=True).expand(-1, -1, J)], dim=-1)
```

> Keep the change minimal: the base PP-correction, positional embeddings, and residual MLP are reused unchanged.

### 3.2 Loss changes

Extend the forward pass to return `log_var`. In the training script, add a Gaussian reprojection NLL auxiliary loss:

```python
nll = 0.5 * (reproj_err_sq * torch.exp(-log_var) + log_var)
loss = mpjpe_loss + uncertainty_loss_weight * nll.mean()
```

Default `uncertainty_loss_weight = 0.1`.

### 3.3 Files to create / modify

| Path | Action |
|---|---|
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_uncertainty_model.py` | New model class `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointUncertainty` |
| `motionflow_mv/fusion/__init__.py` | Register the new model |
| `experiments/train_crossview_residual_principal_point_uncertainty_mpiinf3dhp.py` | New 5-epoch smoke trainer; add `uncertainty_loss_weight` arg |
| `scripts/run_crossview_pp_uncertainty_smoke_wsl.sh` | New shell script to launch the smoke |
| `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` | Add model key so the matrix script can evaluate the new model |

### 3.4 Data / loader

No data or loader change. The model consumes the same `(B, T, V, J, 3)` tensor with confidence in the last channel as the existing PP model.

## 4. Smoke-Test Plan

Run a 3–5 epoch smoke on MPI-INF-3DHP (clean) using the same lightweight config as the factorized smoke:

```bash
python experiments/train_crossview_residual_principal_point_uncertainty_mpiinf3dhp.py \
  --d 32 \
  --residual_hidden 64 \
  --n_st_layers 2 \
  --samples 500 \
  --epochs 5 \
  --batch_size 2 \
  --uncertainty_loss_weight 0.1 \
  --log_var_min -10.0 \
  --log_var_max 10.0
```

**Pass/fail criteria:**

- Pass: val MPJPE finite and ≤ 12.0 mm after 5 epochs (smoke-only threshold; higher than anchor because of tiny config).
- Pass: no NaNs/Inf in `log_var`, weights, or loss; `torch.isnan(loss).any() == False`.
- Pass: predicted `log_var` varies meaningfully across views (std across V > 0.1) on at least 50% of batches.
- Fail: crash, loss divergence, or val MPJPE > 15.0 mm.
- Fail: `log_var` collapses to a constant (std across views < 1e-3), indicating the head is not learning.

## 5. Evaluation Plan

1. **Clean metrics:** run `experiments/eval_full_metrics.py --model uncertainty_pp` on the MPI-INF-3DHP validation split. Target clean MPJPE ≤ 9.6 mm (within 3% of the 9.32 mm anchor) and PA-MPJPE ≤ 5.7 mm.
2. **Robustness matrix:** add the new model key to `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` and run the 6-axis matrix (view dropout, joint dropout, focal, cxcy, rotation, translation). Target: no moderate-severity corruption degrades > 30% relative to clean, and the new model is ≥ 10% better than the PP baseline on `view_dropout_0.4`.
3. **Uncertainty calibration check:** on a held-out 100-clip subset, compare triangulation with/without precision weighting. Target: precision-weighted DLT lowers MPJPE by ≥ 5% versus uniform-weighted DLT when the most uncertain 25% of views are down-weighted.

Scripts:

- `python experiments/eval_full_metrics.py --model uncertainty_pp --checkpoint outputs/crossview_pp_uncertainty_smoke/best.pth`
- `python experiments/eval_robustness_matrix_pp_mpiinf3dhp.py --model uncertainty_pp --n_clips 20`
- `python experiments/compare_uncertainty_calibration.py --model uncertainty_pp --n_clips 100`

## 6. Estimated GPU/CPU Cost on RTX 4090

- Smoke training (5 epochs, 500 samples, d=32, batch=2): ~8–12 minutes on RTX 4090, < 6 GB VRAM.
- Full 30-epoch clean training (if smoke passes): ~2.5–3.5 hours on RTX 4090.
- Robustness matrix (20 clips): ~10 minutes CPU/GPU eval.
- Uncertainty calibration check: ~5 minutes CPU/GPU eval.

Total smoke cost is a single short RTX 4090 job; no CPU-only bottlenecks.

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|---|---|
| Uncertainty head collapses to a constant and provides no benefit | Lower `uncertainty_loss_weight` to 0.01 or remove the auxiliary NLL; keep only the learned weight head |
| Auxiliary NLL destabilizes clean training | Gate the NLL term only after epoch 3, or use curriculum similar to the intrinsics curriculum in the PP trainer |
| Clean accuracy regresses > 3% | Freeze the base PP model weights and train only the new uncertainty head for the first 10 epochs |
| Robustness improvements are only on view dropout, not on intrinsic errors | Add per-view `log_var` conditioned on the corrected intrinsics / PP deltas as extra input features |

---

**Related anchor files:** `docs/results_iter13.md`, `docs/swarm_iter13_next_iteration_synthesis.md`, `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`, `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py`.

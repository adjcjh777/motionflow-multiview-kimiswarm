# v27a: Uncertainty-Aware Depth Proposals

**Task identifier:** `design_v27_uncertainty_depth_proposals`  
**Depends on:** v25 (`docs/proposals/v25_multiview_geometry_fusion.md`), v26 (`docs/proposals/v26_temporal_geometry_fusion.md`)  
**Status:** Design / ready for smoke test

---

## 1. Problem

v25/v26 triangulate each joint with a `DepthProposalTriangulation` head that samples a **fixed, uniform depth grid** (`n_ray_samples=4`, hard-coded `z_min..z_max`) along each viewing ray and scores the candidates (`motionflow_mv/fusion/multiview_geometry_fusion_v25.py`, lines 275–363). This has three concrete weaknesses:

1. **No per-ray adaptivity.** A joint in the far background and a joint near the camera use the same `z_min..z_max` discretisation, so samples are wasted in empty space while the true depth may lie between grid points.
2. **No explicit uncertainty.** The head outputs a single score per candidate but does not model the depth distribution. It cannot express “this ray is ambiguous because the joint is occluded in this view,” which is critical when only 2–4 views are available.
3. **Hard to scale with views.** With 14 views the fixed grid still has only 4 samples per ray; with 2 views the same 4 samples must resolve a much larger ambiguity. Variable-view performance therefore lags at the low-view end.

This proposal directly addresses the v27 decision-matrix item *“Uncertainty-aware depth proposals”* (`docs/proposals/v27_next_iteration_decision_matrix.md`, §2.1).

---

## 2. Proposed method

Replace the fixed depth grid with a **learned per-ray depth distribution**. We keep the rest of the v25/v26 pipeline untouched so the block remains a drop-in, warm-startable upgrade.

### 2.1 New module: `UncertaintyDepthProposalTriangulation`

**File:** `motionflow_mv/fusion/uncertainty_depth_proposal_v27.py`

```text
UncertaintyDepthProposalTriangulation(
    n_views: int = 4,
    n_ray_samples: int = 4,          # now the *number of Monte-Carlo samples*
    n_depth_components: int = 1,        # 1 = single Gaussian, 2+ = mixture
    init_z_min: float = 1.0,
    init_z_max: float = 8.0,
    min_sigma: float = 0.05,          # prevent collapse
    uncertainty_loss_weight: float = 0.01,
)
```

**Inputs / outputs** (drop-in replacement for `DepthProposalTriangulation`):

```python
pred_3d_ref = uncertainty_head(
    centre,        # (B, T, V, 3)
    direction,     # (B, T, V, J, 3)
    confidence,    # (B, T, V, J)
    pred_3d,       # (B, T, J, 3)  current 3D estimate
    view_mask,     # optional (B, T, V)
)
```

Returns `pred_3d_ref` with the same shape `(B, T, J, 3)`.

### 2.2 Per-ray depth distribution

For each `(view, joint)` ray the module predicts:

* `mu_{v,j}` — scalar mean depth.
* `log_sigma_{v,j}` — scalar log-standard-deviation.
* (optional, `n_depth_components > 1`) `mix_logits_{v,j}` — mixture weights.

From these we draw `n_ray_samples` continuous depth values via reparameterisation:

```
eps ~ N(0, 1)
z_s = mu + sigma * eps           # s = 1..n_ray_samples
X_{v,j}^s = c_v + z_s * d_{v,j}
```

At **training** time the samples are stochastic; at **inference** time we use `mu` only (no sampling), so the head is deterministic and fast.

The candidate points `X_{v,j}^s` are scored exactly as in the v25 head, but the score MLP receives the **predicted uncertainty** as an extra input so it can down-weight high-variance rays. Aggregation is still a softmax-weighted average over all `(view, sample)` candidates.

### 2.3 Warm-start / identity property

* The final scoring layer is initialised to zero (as in v25), so scores are uniform at init.
* `mu` is initialised so the mean sample lies at the midpoint of the old `[z_min, z_max]` interval.
* `sigma` is initialised to a large value (≈ `z_max - z_min`), making the initial distribution close to the old uniform grid.
* `residual_scale` starts at `0.0`; the head therefore returns the input `pred_3d` unchanged at the beginning of training.

### 2.4 Extra regularisation

A tiny uncertainty penalty prevents the model from collapsing to zero variance or inflating variance to ignore data:

```
L_unc = uncertainty_loss_weight * mean( |sigma - sigma_target| )
```

where `sigma_target` is a small constant (e.g. `0.2 m`). This is added to the existing `geom_loss` returned by v25/v26.

---

## 3. Integration into the v25/v26 pipeline

### 3.1 Hook location in `omniview_fusion_v5.py`

The new head is used inside the same v25/v26 forward path (`motionflow_mv/fusion/omniview_fusion_v5.py`, lines 787–802). No new hook is needed; we extend the existing `MultiViewGeometryFusionV25` / `TemporalGeometryFusionV26` constructors to instantiate `UncertaintyDepthProposalTriangulation` when a flag is on.

### 3.2 New toggles

Add to `OmniMultiViewFusionV5.__init__` (`motionflow_mv/fusion/omniview_fusion_v5.py`, around line 125–150):

```python
v27_use_uncertainty_depth_proposals: bool = False,
v27_n_depth_components: int = 1,
v27_uncertainty_loss_weight: float = 0.01,
```

Pass them through the v25/v26 module constructors:

```python
self.multiview_geometry_fusion_v25 = MultiViewGeometryFusionV25(
    ...
    use_uncertainty_depth_proposals_v27=v27_use_uncertainty_depth_proposals,
    n_depth_components_v27=v27_n_depth_components,
    uncertainty_loss_weight_v27=v27_uncertainty_loss_weight,
)
```

### 3.3 Changes inside v25/v26

In `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`:

* Lines 275–363 (`DepthProposalTriangulation`) remain untouched; they become the default path.
* Add a new class `UncertaintyDepthProposalTriangulation` in a separate file (`motionflow_mv/fusion/uncertainty_depth_proposal_v27.py`) and import it.
* In `MultiViewGeometryFusionV25.__init__` (lines 385–423), select the head based on `use_uncertainty_depth_proposals_v27`.

In `motionflow_mv/fusion/temporal_geometry_fusion_v26.py`:

* The same selection is applied in `TemporalGeometryFusionV26.__init__` (lines 224–272) because it already re-uses `DepthProposalTriangulation`.

---

## 4. Expected impact

| Metric | Estimate vs. best v25/v26 |
|--------|-----------------------------|
| `val_MPJPE` (H36M / WebBridge) | −5 % to −10 % |
| 2-view MPJPE | −8 % to −15 % (largest gain) |
| 4-view MPJPE | −5 % to −10 % |
| 8-view MPJPE | −3 % to −6 % |
| 14-view MPJPE | −2 % to −4 % |

**Reasoning:** Depth ambiguity is largest when few views are visible; an adaptive depth distribution concentrates samples where the true surface is most likely, directly benefiting the 2/4-view case. With 14 views the DLT seed is already accurate, so gains are smaller but still positive because the head learns to discount occluded rays.

---

## 5. Implementation cost

| Item | Estimate |
|------|----------|
| New lines of code | ~150 in `uncertainty_depth_proposal_v27.py`; ~30 in constructors/flags |
| Modified files | `motionflow_mv/fusion/uncertainty_depth_proposal_v27.py` (new), `multiview_geometry_fusion_v25.py`, `temporal_geometry_fusion_v26.py`, `omniview_fusion_v5.py` |
| Training time increase | +5 % to +10 % (reparameterised sampling adds one extra forward path) |
| Memory increase | negligible (`n_ray_samples` stays at 4) |
| Data needs | none beyond current v25/v26 mix |
| Test additions | `tests/test_uncertainty_depth_proposal_v27.py` (~150 lines) |

---

## 6. Risks / mitigation

| Risk | Detection | Mitigation |
|------|-----------|------------|
| Uncertainty collapses to near-zero, overfitting to training depth range | Monitor histogram of predicted `sigma` per epoch | `min_sigma` clamp + `L_unc` regulariser; if mean `sigma < 0.03`, raise `min_sigma` |
| Sampling variance makes training noisy | Loss curve jitters; gradient norm spikes | Use `n_ray_samples=1` during first epoch, increase to 4 later; or use straight-through estimator at inference |
| Single Gaussian is too restrictive | Little MPJPE improvement at 2 views | Increase to `n_depth_components=2` or 4 |
| v25 checkpoint incompatibility | Loading old checkpoint raises missing-key warnings | Prefix new parameters so old checkpoints load with default `use_uncertainty_depth_proposals_v27=False` |
| Inference determinism | Smoke test output differs run-to-run | At inference use `mu` only; add `torch.manual_seed` guard in unit tests |

---

## 7. Minimal experiment plan

### 7.1 Flags / config names

Use these in the training YAML / model kwargs:

```yaml
v27_use_uncertainty_depth_proposals: true
v27_n_depth_components: 1
v27_uncertainty_loss_weight: 0.01
v25_use_learned_depth_triangulation: true   # keep the depth head enabled
```

### 7.2 Smoke test

Run the unit test on CPU/GPU before any full training:

```bash
# Add and run the v27 test once the module exists
PYTHONPATH=. .venv/bin/pytest tests/test_uncertainty_depth_proposal_v27.py -q
```

Then run a one-epoch smoke on the local RTX 4090:

```bash
python experiments/run_train.py \
  config=configs/train_focal_calibration_smoke.yaml \
  model.use_multiview_geometry_fusion_v25=true \
  model.v25_use_learned_depth_triangulation=true \
  model.v27_use_uncertainty_depth_proposals=true \
  model.v27_n_depth_components=1 \
  trainer.max_epochs=1
```

### 7.3 Decision gate

Compare against the matching v25 small baseline using the same seed/data split:

* If `val_MPJPE` improves by ≥ 2 mm, launch a full A800 run.
* If 2-view MPJPE improves but 14-view degrades, keep only the single-Gaussian variant and reduce `v27_uncertainty_loss_weight`.
* If no improvement after two epochs, stop and try `n_depth_components=2` before discarding the direction.

---

## 8. Simpler variant (if the full mixture is too vague)

Start with the **single-Gaussian, continuous-sampling version only** (`n_depth_components=1`). It is a ~120-line change, fully warm-starts from v25, and gives most of the expected gain without the risk of a mixture collapsing to one component. The mixture extension can be added later if the single-Gaussian variant plateaus.

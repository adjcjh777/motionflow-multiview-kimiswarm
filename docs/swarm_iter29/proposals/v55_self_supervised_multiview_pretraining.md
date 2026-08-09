# v55 Self-Supervised Multi-View Pretraining (SSMP)

## 1. Module name and one-line purpose

**Module:** `SelfSupervisedMultiViewPretrainingV55` → `motionflow_mv/fusion/self_supervised_multiview_pretraining_v55.py`

**One-line purpose:** Refine the v54 physically calibrated 3-D pose with masked-view triangulation consistency, cross-view feature agreement, and a temporal smoothness prior, all gated so the module is identity-at-init.

## 2. Position in the OmniMultiViewFusionV5 forward pass

Placed **after** `PhysicalSpaceCalibrationV2V54` (v54 PSC-v2) and **before** the final residual MLP / v47/v49 temporal / v50 SEFH heads.

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, geo_features
    ↓
v52 UWT → pred_3d_uwt, uwt_weights
    ↓
v53 PSC → pred_3d_psc
    ↓
v54 PSC-v2 → pred_3d_psc2
    ↓
v55 SelfSupervisedMultiViewPretrainingV55
    (consumes pred_3d_psc2, points_2d, confidences, K, R, t, view_mask, geo_features)
    → pred_3d_ssmp, ssmp_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

v55 treats multi-view consistency as an auxiliary refinement objective: it masks views during training, asks the network to reproduce the masked observations from the unmasked subset, and adds a gated residual correction to the pose.

## 3. Inputs, outputs, and shapes

**Inputs**

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_psc2` | `(B, T, J, 3)` | Calibrated 3-D pose from v54 PSC-v2. |
| `points_2d` | `(B, T, J, V, 2)` | Input 2-D keypoints. |
| `confidences` | `(B, T, J, V)` | Per-keypoint detection confidences. |
| `K` | `(B, T, V, 3, 3)` | Camera intrinsics. |
| `R` | `(B, T, V, 3, 3)` | Camera rotations. |
| `t` | `(B, T, V, 3)` | Camera translations. |
| `view_mask` | `(B, T, V)` | Binary valid-view mask. |
| `geo_features` | `(B, T, J, V, D)` | Geometry-fusion features from v25/v45 (reused for cross-view consistency). |
| `domain_id` | `(B,)` or `(B, T)` | Domain label for per-domain behavior (optional, passed through). |

**Outputs**

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_ssmp` | `(B, T, J, 3)` | Refined 3-D pose. Identity-at-init means `pred_3d_ssmp ≈ pred_3d_psc2`. |
| `ssmp_loss` | scalar | Auxiliary self-supervised loss (reprojection + consistency + temporal). |

## 4. Architecture

### 4.1 Gated residual pose correction

```
gate = sigmoid(gate_logit)          # gate_logit init = -6.0  → gate ≈ 0.0025
delta = MLP([flatten(pred_3d_psc2, geo_pooled)]; hidden=v55_ssmp_hidden, layers=v55_ssmp_n_layers)
pred_3d_ssmp = pred_3d_psc2 + gate * delta
```

- The final layer of the `delta` MLP is zero-initialized.
- `gate_logit` is a learnable scalar initialized to `v55_ssmp_residual_gate_init = -6.0`.
- Therefore at init `pred_3d_ssmp = pred_3d_psc2 + 0.0025 * 0 ≈ pred_3d_psc2`.

### 4.2 Masked-view triangulation head

During training only:
1. Sample a random subset of views to mask: `masked_views ~ Bernoulli(p=v55_ssmp_mask_prob)` subject to `v55_ssmp_min_views` remaining unmasked.
2. Triangulate `pred_3d_psc2` using only the **unmasked** views to obtain `pred_3d_masked`.
3. Reproject `pred_3d_masked` to the masked views and compute a soft reprojection loss weighted by original confidence:
   ```
   L_reproj = Σ_masked_views ||π_v(pred_3d_masked) - points_2d[v]||² * confidence[v]
   ```
4. Triangulate again using all views to get `pred_3d_full`.
5. Consistency loss: `L_tri_cons = ||pred_3d_masked - pred_3d_full.detach()||²` (encourages masked-view triangulation to agree with full-view triangulation).

### 4.3 Cross-view feature consistency head

1. Pool per-view geometry features `geo_features` across joints using confidences to get per-view descriptors `h_v ∈ R^D`.
2. Pass the set `{h_v}` through a small transformer (1 layer, `v55_ssmp_hidden`, 4 heads) to produce view-aligned descriptors `h'_v`.
3. Predict a per-view consistency score:
   ```
   c_v = MLP([h_v, h'_v])   # zero-init final layer
   ```
4. Consistency loss matches the predicted score to the reprojection error of that view:
   ```
   L_cons = Σ_v (c_v - e_v)²
   ```
   where `e_v` is the per-view reprojection error of `pred_3d_ssmp`. This head is identity-at-init because `c_v ≈ 0` and its gradient to the pose is zero until the scores learn.

### 4.4 Temporal smoothness head

On clips (`T > 1`):
```
L_temp = Σ_{t=1}^{T-1} ||pred_3d_ssmp[t] - pred_3d_ssmp[t-1]||² * exp(-λ |pred_3d_psc2[t] - pred_3d_psc2[t-1]|²)
```
The velocity-dependent weighting prevents over-smoothing of fast motion.

### 4.5 Total auxiliary loss

```
ssmp_loss = v55_ssmp_reproj_weight   * L_reproj
          + v55_ssmp_consistency_weight * L_cons
          + v55_ssmp_temporal_weight   * L_temp
          + v55_ssmp_tri_consistency_weight * L_tri_cons
```

Multiplied by `v55_ssmp_loss_weight` in the trainer and added only after `v55_ssmp_warmup_epochs`.

### 4.6 Identity-at-init mechanism

| Component | Init / behavior |
|---|---|
| `delta` MLP final layer | Zero init → residual `δ = 0`. |
| `gate_logit` | `-6.0` → `gate ≈ 0.0025`. |
| Consistency score MLP final layer | Zero init → `c_v = 0`. |
| Warm-up | `ssmp_loss` weight is zero for `v55_ssmp_warmup_epochs` epochs. |

## 5. Expected MPJPE impact and main risks

**Expected impact**

- Full views: `−0.5 to −1.5 mm` by reducing over-reliance on noisy individual views and improving triangulation consistency.
- Sparse views (`@2/3`): `3–6%` relative improvement; masking during training directly simulates sparse-view conditions, which should improve `@2` and `@3` more than full views.
- Temporal stability: small gain on fast-motion sequences.

**Main risks and mitigations**

| Risk | Symptom | Mitigation |
|---|---|---|
| Masking too many views destroys triangulation signal | `L_reproj` explodes, NaN/Inf | Cap mask probability at `0.3`; enforce `min_views=2`; clip reprojection residuals. |
| Cross-view consistency head overfits to training camera setups | Validation MPJPE rises | Keep transformer tiny (1 layer, hidden `64`); use dropout `0.1`. |
| Temporal smoothness over-smooths fast motion | Large errors on jumping/running | Velocity-dependent weighting; make `L_temp` optional via flag. |
| Identity-at-init fails | v54 checkpoint regresses `>0.1 mm` with v55 enabled | Unit test `||pred_3d_ssmp - pred_3d_psc2||_∞ < 1e-4`; zero-init final layers; gate logit `-6.0`. |
| Auxiliary loss dominates early training | Total loss spikes, training destabilizes | Warm-up `v55_ssmp_warmup_epochs=1`; default loss weight `0.01`. |

## 6. Smoke acceptance criteria (RTX 4090)

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- `val_MPJPE@2` and `val_MPJPE@3` are not worse than the v54 baseline.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Masked-view sanity: at most `30%` of views are masked and at least `2` views remain unmasked in every training batch.
- Reprojection sanity: `L_reproj` is finite and decreasing over the first epoch.
- Cross-view consistency sanity: predicted consistency scores `c_v` stay in `[-5, 5]` (bounded by `tanh`).

## 7. Required new files and files to modify

**New files**

- `motionflow_mv/fusion/self_supervised_multiview_pretraining_v55.py` — new module containing `SelfSupervisedMultiViewPretrainingV55`.
- `configs/benchmark_v55_ssmp_smoke.yaml` — smoke config copied from the v54 PSC-v2 smoke config with v55 flags enabled.
- `scripts/run_v55_ssmp_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v54 checkpoint.
- `tests/test_self_supervised_multiview_pretraining_v55.py` — unit tests for identity-at-init, masking logic, reprojection, and gradient flow.

**Files to modify**

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add the v55 flag block in `__init__`.
  - Instantiate `SelfSupervisedMultiViewPretrainingV55` when enabled.
  - Insert the call after the v54 PSC-v2 block.
  - Add `ssmp_loss` to the `epi_loss` dictionary under key `v55_ssmp`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
 - Forward `domain_id` to the model.
 - Aggregate `loss_dict["v55_ssmp"]` with weight `v55_ssmp_loss_weight` only after `v55_ssmp_warmup_epochs`.
- `scripts/launch_v33_a800_queue.py`
 - Add an A800 full-run entry for v55 SSMP on top of the best v54 checkpoint.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_self_supervised_multiview_pretraining` | bool | `False` | Master toggle. |
| `v55_ssmp_hidden` | int | `64` | Hidden dimension of MLP/transformer heads. |
| `v55_ssmp_n_layers` | int | `2` | Depth of the residual MLP. |
| `v55_ssmp_mask_prob` | float | `0.30` | Probability of masking each view during training. |
| `v55_ssmp_min_views` | int | `2` | Minimum unmasked views required. |
| `v55_ssmp_identity_init` | bool | `True` | Zero-initialize final residual layers and gate. |
| `v55_ssmp_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init. |
| `v55_ssmp_use_masked_view_loss` | bool | `True` | Enable masked-view reprojection loss. |
| `v55_ssmp_use_cross_view_consistency` | bool | `True` | Enable cross-view feature consistency head. |
| `v55_ssmp_use_temporal_consistency` | bool | `True` | Enable velocity-weighted temporal smoothness loss. |
| `v55_ssmp_loss_weight` | float | `0.01` | Multiplier on total `ssmp_loss`. |
| `v55_ssmp_reproj_weight` | float | `1.0` | Weight of `L_reproj`. |
| `v55_ssmp_consistency_weight` | float | `0.1` | Weight of `L_cons`. |
| `v55_ssmp_temporal_weight` | float | `0.01` | Weight of `L_temp`. |
| `v55_ssmp_tri_consistency_weight` | float | `0.1` | Weight of `L_tri_cons`. |
| `v55_ssmp_warmup_epochs` | int | `1` | Epochs before `ssmp_loss` contributes. |

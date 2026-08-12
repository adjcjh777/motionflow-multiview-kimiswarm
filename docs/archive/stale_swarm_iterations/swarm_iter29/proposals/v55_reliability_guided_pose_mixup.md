# v55 Reliability-Guided Pose Mixup

## 1. Module name and one-line purpose

**Module:** `ReliabilityGuidedPoseMixupV55` → `motionflow_mv/fusion/reliability_guided_pose_mixup_v55.py`

**One-liner:** After v54’s local physical calibration, softly blend each joint’s 3-D pose with a learned per-domain canonical anchor, where the blending strength is conditioned on the v52 uncertainty-weighted triangulation reliability; high-reliability joints keep the calibrated pose, while low-reliability / sparse-view joints are nudged toward a plausible canonical shape.

## 2. Forward-pass location

`ReliabilityGuidedPoseMixupV55` sits **after** `PhysicalSpaceCalibrationV2V54` (v54 PSC-v2) and **before** the final residual MLP / v47/v49 temporal / v50 SEFH heads.

```textnpoints_2d, confidences, K, R, t
    
v25/v45 geometry fusion → pred_3d_init, weights_initn    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_lossn    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_lossn    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_lossn    ↓
v55 ReliabilityGuidedPoseMixupV55
    (consumes pred_3d_psc2, uwt_weights, domain_id)
    → pred_3d_rgpm, rgpm_loss, alpha
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

## 3. Inputs, outputs, and shapes

### Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Calibrated 3-D pose from v54 PSC-v2. |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view-per-joint triangulation weights. |
| `domain_id` | `(B,)` | Domain index for per-domain canonical anchors. |
| `view_mask` | `(B, T, V)` | Optional visibility mask for computing per-joint reliability. |

Derived input:

- `reliability`: `(B, T, J)` in `[0, 1]`, obtained by max-pooling `uwt_weights` over views and applying a temperature-scaled softmax/sigmoid normalization. Missing views are masked out before pooling.

### Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d_rgpm` | `(B, T, J, 3)` | Reliability-mixed 3-D pose. Identity-at-init returns `pred_3d` unchanged. |
| `rgpm_loss` | `scalar` | Auxiliary anchor-consistency + alpha-sparsity loss. |
| `alpha` | `(B, T, J)` | Per-joint mixing strength in `[0, 1]`. |

## 4. Architecture

### Core blocks

1. **Per-joint reliability aggregator**
   - Input: `uwt_weights` `(B, T, V, J)` and `view_mask` `(B, T, V)`.
   - Output: `reliability` `(B, T, J)`.
   - Operation: mask out missing views, take max over visible views, then pass through a learned 1-D projection (linear → sigmoid) initialized so that the output is approximately the raw max weight. Final layer of the projection is zero-initialized.

2. **Per-domain canonical anchor**
   - Parameter: `canonical_pose` of shape `(num_domains, J, 3)` if `v55_rgpm_use_domain_anchor=True`, otherwise `(J, 3)`.
   - Initialization: zeros. During training it learns a dataset-level mean pose per domain.

3. **Anchor refiner MLP**
   - Input: flattened per-joint `[canonical_pose[j], reliability_j]` plus a learned joint embedding.
   - Hidden: `v55_rgpm_hidden` (default `64`), 1–2 layers.
   - Output: per-joint 3-D offset.
   - **Identity-at-init:** the final output layer is zero-initialized, so `anchor_eff = canonical_pose` at start.

4. **Mixing-coefficient head**
   - Input: per-joint reliability + joint embedding.
   - Architecture: 2-layer MLP, hidden `v55_rgpm_hidden`, output scalar logit.
   - `alpha = sigmoid(MLP(reliability) + v55_rgpm_gate_init)`, where `v55_rgpm_gate_init = -6.0` gives `alpha ≈ 0.0025` at init.
   - **Identity-at-init:** `alpha ≈ 0`, so `pred_3d_rgpm = pred_3d` with no effective change.

5. **Pose mixup**
   ```text
   pred_3d_rgpm = (1 - alpha[..., None]) * pred_3d + alpha[..., None] * anchor_eff
   ```

### Losses

| Loss | Definition | Weight |
|------|------------|--------|
| `L_anchor` | `mean( reliability * ||pred_3d - anchor_eff||² )` | `v55_rgpm_anchor_weight` (default `0.01`) |
| `L_alpha` | `mean(alpha²)` | `v55_rgpm_alpha_l2_weight` (default `0.001`) |

Total: `rgpm_loss = v55_rgpm_loss_weight * (L_anchor + L_alpha)`.

The anchor loss pulls the learned canonical anchor toward high-reliability joints, while the alpha L2 penalty keeps the mixing conservative until the data demands it. Both terms are zero or negligible at initialization.

## 5. Expected MPJPE impact and main risks

### Expected impact

- **Full views:** `−0.2` to `−0.7 mm`. Most joints already have high v52 reliability, so the mixup has little effect; gains come from subtle regularization of outliers and rare low-confidence joints.
- **Sparse views (`@2`/ `@3`):** `−1.0` to `−2.5 mm`. Low-reliability joints in sparse-view settings benefit the most from being pulled toward a learned canonical shape.
- **Identity-at-init:** `< 0.1 mm` change when loading a v54 checkpoint with v55 enabled before any training step.

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **Over-smoothing** | Wrists/ankles collapse toward the mean pose; MPJPE rises. | Clamp `alpha ≤ 0.5` during training, initialize gate to `-6.0`, add `L_alpha` penalty, keep hidden dim small (`64`). |
| **Anchor collapse** | Canonical anchor drifts far from plausible human shapes. | Anchor loss is weighted by reliability, so only high-confidence joints supervise the anchor; also clip anchor magnitude per joint. |
| **Domain confusion** | Anchor mixes proportions across H36M / MPI / 3DPW. | Per-domain `canonical_pose`; use `domain_id` from v48 domain generalization if available. |
| **Identity-at-init failure** | Loading v54 checkpoint with v55 changes `val_MPJPE` by `>0.1 mm`. | Zero-init all final output layers and the gate bias; unit-test `||pred_3d_rgpm - pred_3d||_∞ < 1e-4` at init. |

## 6. Smoke acceptance criteria

1. **Baseline preservation:** `val_MPJPE@full` is within `1 mm` of the v54 PSC-v2 baseline on the same smoke config.
2. **Identity-at-init:** loading the best v54 checkpoint with v55 enabled and taking **no training step** changes `val_MPJPE` by `< 0.1 mm`.
3. **Stability:** no NaN, Inf, or OOM through at least one full smoke epoch.
4. **Conservative mixing:** at least `95%` of `alpha` values are `< 0.1` at initialization, and after the first epoch `max(alpha) < 0.5`.
5. **Sparse-view non-regression:** `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline (within `0.5 mm`).
6. **Anchor sanity:** per-joint canonical anchors remain finite and do not drift more than `1 m` from the training-set mean pose after one smoke epoch.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/reliability_guided_pose_mixup_v55.py` — `ReliabilityGuidedPoseMixupV55` module.
- `configs/benchmark_v55_reliability_guided_pose_mixup_smoke.yaml` — smoke config copied from `configs/benchmark_v54_psc_v2_smoke.yaml` with v55 flags enabled.
- `scripts/run_v55_reliability_guided_pose_mixup_smoke_local_4090.sh` — smoke launch script that warm-starts from the best v54 checkpoint.
- `tests/test_reliability_guided_pose_mixup_v55.py` — unit tests for identity-at-init, per-domain anchor shape, alpha bounds, and gradient flow.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add the v55 flag block in `__init__`, instantiate the module when enabled, call it after v54 PSC-v2, and add `rgpm_loss` to the existing `epi_loss` dictionary under key `v55_rgpm`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — ensure `domain_id` is forwarded and aggregate `loss_dict["v55_rgpm"]` with `v55_rgpm_loss_weight` after `v55_rgpm_warmup_epochs`.
- `scripts/launch_v33_a800_queue.py` — add an A800 full-run entry for `v55_reliability_guided_pose_mixup_on_v54`.
- `AGENTS.md` — append a short v55 conventions section with flags and workflow.

### Config flags and defaults

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v55_reliability_guided_pose_mixup` | bool | `False` | Master toggle |
| `v55_rgpm_hidden` | int | `64` | Hidden dimension of anchor/mixing MLPs |
| `v55_rgpm_num_domains` | int | `8` | Number of domains for per-domain anchors |
| `v55_rgpm_use_domain_anchor` | bool | `True` | Use per-domain canonical anchors |
| `v55_rgpm_identity_init` | bool | `True` | Zero-initialize final output layers |
| `v55_rgpm_gate_init` | float | `-6.0` | Mixing gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_rgpm_anchor_weight` | float | `0.01` | Weight of `L_anchor` |
| `v55_rgpm_alpha_l2_weight` | float | `0.001` | Weight of `L_alpha` |
| `v55_rgpm_loss_weight` | float | `1.0` | Multiplier on total `rgpm_loss` |
| `v55_rgpm_warmup_epochs` | int | `0` | Epochs before `rgpm_loss` contributes to total loss |
| `v55_rgpm_max_alpha` | float | `0.5` | Hard upper bound on `alpha` during training |

---

**Tracking issue:** #189 (proposed)  
**Base branch:** `v55-reliability-guided-pose-mixup`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

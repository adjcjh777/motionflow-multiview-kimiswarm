# v55 Adaptive Low-Rank Per-Domain Fusion (ALRPD)

**Module:** `AdaptiveLowRankPerDomainV55` → `motionflow_mv/fusion/adaptive_low_rank_per_domain_v55.py`  
**Branch:** `v55-alrpd`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2  
**Tracking issue:** TBD (#185 expected)

---

## 1. Module name and one-line purpose

`AdaptiveLowRankPerDomainV55` removes residual domain-specific pose bias after physical-space calibration by applying a **gated, low-rank per-domain affine adapter** to the calibrated 3-D pose, conditioned on v52 uncertainty weights and v54 physical features.

---

## 2. Position in the `OmniMultiViewFusionV5` forward pass

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2
    ↓
v55 AdaptiveLowRankPerDomainV55
    (consumes pred_3d_psc2, uwt_weights, psc2_features, domain_id)
    → pred_3d_alrpd, alrpd_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

The module sits **immediately after v54 PSC-v2** and **before any final residual MLP or temporal/SEFH heads**. It therefore refines the already physically calibrated pose with a lightweight domain-specific correction, leaving all earlier blocks untouched.

---

## 3. Inputs, outputs, and shapes

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_psc2` | `(B, T, J, 3)` | Calibrated 3-D pose from v54. |
| `domain_id` | `(B,)` | Integer domain label per clip (H36M=0, MPI=1, WebBridge=2, 3DPW=3, etc.). |
| `uwt_weights` | `(B, T, V, J)` | v52 per-(view, joint) uncertainty weights. |
| `psc2_features` | `(B, T, J, C)` | Optional physical features from v54 (floor distance, bone-scale residual, reprojection error). If absent, the module computes per-joint features from `pred_3d_psc2`. |
| `view_mask` | `(B, T, V)` | Optional mask for valid views. |

**Outputs:**

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_alrpd` | `(B, T, J, 3)` | Domain-adapted 3-D pose. |
| `alrpd_loss` | `scalar` | Auxiliary regularization on the low-rank factors. |

---

## 4. Architecture

### 4.1 Per-joint feature extractor

For each joint `j` and timestep `t`, compute a feature vector:

```
f_j = LayerNorm(Linear(3 → hidden))(pred_3d_psc2[:, t, j, :])   # (B, hidden)
```

If `psc2_features` is provided, concatenate it to `f_j` and project back to `hidden`.

### 4.2 Per-domain low-rank adapters

Maintain a learnable lookup table of domain-specific low-rank factors:

```
A_d ∈ R^{hidden × rank}   (random init, scaled by 1/sqrt(rank))
B_d ∈ R^{hidden × rank}   (zero-initialized)
```

For a given domain `d = domain_id[b]`:

```
M_d = B_d @ A_d^T                                    # (hidden × hidden)
Δf_j = M_d @ f_j                                      # (B, hidden)
```

### 4.3 Uncertainty-weighted gating

Aggregate v52 weights per joint to form a reliability scalar:

```
w_j = mean_v(uwt_weights[:, :, v, j])                 # (B, T, J)
gate_logit = learned_scalar(init = -6.0)             # scalar
gate = sigmoid(gate_logit) * clamp(w_j, min=0.05, max=1.0)
```

The uncertainty gate down-weights joints with high triangulation uncertainty, keeping the correction conservative when views are sparse or unreliable.

### 4.4 Residual pose correction

Project the adapted feature back to 3-D and add a gated residual:

```
Δpose_j = Linear(hidden → 3)(Δf_j)                    # (B, T, J, 3)
pred_3d_alrpd = pred_3d_psc2 + gate * Δpose_j
```

### 4.5 Auxiliary loss

```
alrpd_loss = v55_alrpd_reg_weight * ( ||B_d||_F^2 + ||A_d||_F^2 / rank )
```

The regularization is divided by `rank` to keep the magnitude of `A_d` comparable across ranks and to prevent the factors from growing during warm-start.

### 4.6 Identity-at-init mechanism

- `B_d` is zero-initialized, so `M_d = 0` at initialization.
- The output projection `Linear(hidden → 3)` is zero-initialized.
- `gate_logit` is initialized to `-6.0`, giving `sigmoid(gate) ≈ 0.0025`.
- Therefore `pred_3d_alrpd = pred_3d_psc2` and a v54 checkpoint loads unchanged.

---

## 5. Expected MPJPE impact and main risks

### Expected impact

| View setting | Expected change |
|---|---|
| Full views | `-0.3 to -0.8 mm` |
| `@4` | `-0.5 to -1.0 mm` |
| `@3` | `-1.0 to -2.0 mm` |
| `@2` | `-1.0 to -2.5 mm` |
| Cross-domain (3DPW actual) | `-0.8 to -2.0 mm` |

The largest gains are expected on sparse-view and cross-domain evaluation, where domain-specific residual bias (e.g., camera/studio vs. in-the-wild capture characteristics) is most pronounced.

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| **Domain overfitting** | In-domain improves, 3DPW/cross-domain regresses. | Keep rank small (`rank ≤ 8`), add `alrpd_loss` regularization, and mask out domains with too few samples. |
| **Rank too small/large** | No gain or unstable gradients. | Default `rank=4`; provide `rank=2` and `rank=8` ablations. |
| **Conflicts with v48 domain generalization** | v48 FiLM/GRL already removes domain bias; ALRPD may double-correct. | Make the module optional, initialize conservatively, and ablate with `use_v48_domain_generalization=False`. |
| **Identity-at-init regression** | v54 checkpoint changes by `>0.1 mm` when ALRPD is enabled. | Zero-init `B_d`, output projection, and gate logit; unit-test `||pred_alrpd - pred_psc2||_∞ < 1e-4`. |
| **Memory overhead from per-domain factors** | O(num_domains × hidden × rank) extra parameters. | Default `num_domains=8`, `hidden=64`, `rank=4` → only ~2 k extra parameters. |

---

## 6. Smoke acceptance criteria

1. **Baseline preservation:** loading the best v54 checkpoint with `use_v55_adaptive_low_rank_per_domain=True` and no training step changes `val_MPJPE@full` by `< 0.1 mm`.
2. **No regression:** after one smoke epoch, `val_MPJPE@full` is within `1 mm` of the v54 baseline.
3. **Stability:** no NaN, Inf, or OOM through at least one full epoch on RTX 4090.
4. **Factor sanity:** `||B_d||_F` stays below `1.0` and `||A_d||_F / sqrt(rank)` stays below `5.0` for all domains.
5. **Sparse-view non-regression:** `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.
6. **Cross-domain sanity:** per-domain `MPJPE` remains finite for all seen domains.

---

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/adaptive_low_rank_per_domain_v55.py` — module implementing `AdaptiveLowRankPerDomainV55`.
- `configs/benchmark_v55_adaptive_low_rank_per_domain_smoke.yaml` — smoke config copied from v54 with v55 flags enabled.
- `scripts/run_v55_adaptive_low_rank_per_domain_smoke_local_4090.sh` — smoke launch script that warm-starts from the best v54 checkpoint.
- `tests/test_adaptive_low_rank_per_domain_v55.py` — unit tests for identity-at-init, per-domain factor shapes, factor regularization, and gradient flow.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add constructor flag `use_v55_adaptive_low_rank_per_domain`.
  - Instantiate `AdaptiveLowRankPerDomainV55` when enabled.
  - Insert the call after the v54 PSC-v2 block and before the final residual MLP / v47/v49 / v50 heads.
  - Add `alrpd_loss` to the `epi_loss` dictionary under key `v55_alrpd`.

- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Forward `domain_id` into the model.
  - Aggregate `loss_dict["v55_alrpd"]` with `v55_alrpd_loss_weight` after any warmup epochs.

- `scripts/launch_v33_a800_queue.py`
  - Add A800 full-run entry `v55_adaptive_low_rank_per_domain_on_v54`.

### Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_adaptive_low_rank_per_domain` | bool | `False` | Master toggle |
| `v55_alrpd_hidden` | int | `64` | Per-joint feature dimension |
| `v55_alrpd_rank` | int | `4` | Low-rank adapter rank |
| `v55_alrpd_num_domains` | int | `8` | Number of learned domain adapters |
| `v55_alrpd_identity_init` | bool | `True` | Zero-initialize `B_d` and output projection |
| `v55_alrpd_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_alrpd_use_uwt_gate` | bool | `True` | Scale correction by per-joint v52 reliability |
| `v55_alrpd_loss_weight` | float | `1.0` | Multiplier on `alrpd_loss` |
| `v55_alrpd_reg_weight` | float | `0.01` | Weight of Frobenius regularization on factors |
| `v55_alrpd_warmup_epochs` | int | `0` | Epochs before `alrpd_loss` contributes to total loss |

---

## 8. Paper alignment

ALRPD extends the physical-space calibration story into the **domain adaptation** stage. After v53/v54 make the pose physically consistent, the remaining errors are often dataset-specific capture biases (camera height, subject population, marker placement). A low-rank per-domain adapter is a minimal, interpretable, and parameter-efficient way to absorb those biases without changing the shared pose representation, aligning with the paper narrative of *multi-view fusion → physical calibration → domain-robust motionflow*.

# v52: Outlier-Robust Reliability (ORR)

**Author:** design-swarm agent
**Module key:** `outlier_robust_reliability_v52`
**Depends on:** v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, v25/v45 geometry fusion, v28 physical-space alignment
**Tracking issue:** `#182`

---

## 1. Motivation

The current pipeline (v50 SEFH → v51 CDSVR) already closes the *pose → reliability → uncertainty* loop, but it treats reprojection/epipolar residuals as plain L2 signals. A single occluded or badly calibrated view can therefore dominate the per-joint uncertainty because outliers are not explicitly bounded. v52 introduces an **outlier-robust reliability head** that:

1. Uses a robust M-estimator instead of raw squared residuals when scoring view consistency.
2. Fuses **physical-space alignment cues** (bone-length consistency and floor penetration) into the reliability update.
3. Remains **warm-startable / identity-at-init**, so it can be dropped on top of a trained v50/v51 checkpoint without regression.

This directly strengthens the *multi-view fusion and calibration* → *physical-space alignment* → *optimized motionflow pipeline* story in the paper: the model learns to treat geometrically and physically inconsistent views as outliers, producing cleaner triangulations before the physical-space stage.

---

## 2. Architecture

### 2.1 Where it lives in `OmniMultiViewFusionV5`

The module is instantiated when `use_v52_outlier_robust_reliability=True`. It is called **after** the v50 SEFH returns `(reliability_v50, log_var_v50, reproj, temporal, epipolar, hidden)` and **before** the weighted triangulation / physical-space alignment blocks:

```text
v25/v45 geometry fusion
        ↓
v50 SEFH → v51 CDSVR → (r_v51, σ_v51)
        ↓
v52 ORR (this module)  → (r_v52, σ_v52)
        ↓
weighted DLT / triangulation
        ↓
physical-space alignment (v28/v40)
```

### 2.2 Internal blocks

| Block | Purpose | Output shape |
|---|---|---|
| `RobustResidualEncoder` | Apply Huber/Tukey M-estimator to residuals; clip large outliers. | `(B, T, V, J, 4)` |
| `PhysicalCueEncoder` | Encode bone-length residual and floor-penetration signal per joint. | `(B, T, V, J, 2)` |
| `ReliabilityRefiner` | Cross-view attention over (view, joint) tokens; predicts additive offset. | `(B, T, V, J)` |

All three blocks are zero-initialized at their final projection layers so that at init `r_v52 ≈ r_v51` and `σ_v52 ≈ σ_v51`.

### 2.3 Equations

Let

- `r ∈ [0,1]^{B×T×V×J}`: per-view per-joint reliability from v51 CDSVR.
- `σ ∈ R_{>0}^{B×T×J}`: per-joint uncertainty from v51 CDSVR.
- `e_reproj, e_epi ∈ R^{B×T×V×J}`: reprojection and epipolar residuals from v50 SEFH.
- `e_bone, e_floor ∈ R^{B×T×J}`: physical-space residuals signals (bone-length deviation, floor penetration depth).

**Robust M-estimator (Tukey bisquare variant):**

```
ρ_c(x) = (c²/6) · (1 - [1 - (x/c)²]³)      if |x| ≤ c
         (c²/6)                            if |x| > c
w_c(x) = ∂ρ/∂x / x = [1 - (x/c)²]²          if |x| ≤ c
         0                                   if |x| > c
```

with `c = v52_residual_clip_percentile` (default 0.95 over-batch). The robust weight is `w_geo = w_c(e_reproj) · w_c(e_epi)`.

**Physical cue encoding:**

```
bone_view  = broadcast(e_bone, V)                  # (B,T,V,J)
floor_view = broadcast(e_floor, V)                 # (B,T,V,J)
phys_feat  = concat([log(1+e_bone), log(1+e_floor)])  # (B,T,V,J,2)
```

**Reliability refinement:**

```
f_robust = RobustResidualEncoder(concat([e_reproj, e_epi, w_geo]))  # (B,T,V,J,4)
q      = ReliabilityRefiner(concat([f_robust, phys_feat]))        # (B,T,V,J)
r_v52  = sigmoid( logit(r_v51) + α · q )                           # (B,T,V,J)
σ_v52  = σ_v51 / (1 + β · TanhGate( pooled_phys ))                 # (B,T,J)
```

where `α, β` are learnable scalar gains initialized to 0.0 (identity). `TanhGate` is a small MLP with `tanh` output so uncertainty shrinks only when physical cues agree across views.

---

## 3. Inputs and Outputs

### Inputs to `OutlierRobustReliabilityV52`

| Symbol | Shape | Description |
|---|---|---|
| `reliability_v51` | `(B, T, V, J)` | Reliability from v51 CDSVR. |
| `log_var_v51` | `(B, T, J)` | Log-variance from v51 CDSVR. |
| `reproj_residual` | `(B, T, V, J)` | v50 SEFH reprojection residual. |
| `epipolar_residual` | `(B, T, V, J)` | v50 SEFH epipolar residual. |
| `pred_3d` | `(B, T, J, 3)` | Current 3-D pose estimate. |
| `camera_params` | `(K, R, t)` | Camera intrinsics/extrinsics. |
| `domain_emb` | `(B, d)` or `None` | Optional domain embedding. |
| `view_mask` | `(B, T, V)` or `None` | Active-view mask. |

### Outputs

| Symbol | Shape | Description |
|---|---|---|
| `reliability_v52` | `(B, T, V, J)` | Refined per-view reliability in `[v52_min_rel, 1]`. |
| `log_var_v52` | `(B, T, J)` | Refined per-joint log-variance. |
| `robust_weight` | `(B, T, V, J)` | Diagnostic robust M-estimator weight. |

---

## 4. Config Flags

```yaml
use_v52_outlier_robust_reliability: false
v52_orr_hidden: 64
v52_orr_num_heads: 4
v52_orr_num_layers: 2
v52_orr_dropout: 0.1
v52_orr_identity_init: true
v52_orr_min_rel: 0.05
v52_orr_m_estimator: "tukey"        # {"huber", "tukey"}
v52_orr_residual_clip_percentile: 0.95
v52_orr_alpha_init: 0.0             # gain on reliability offset
v52_orr_beta_init: 0.0             # gain on uncertainty rescale
v52_orr_use_physical_cues: true
v52_orr_use_domain_conditioning: true
v52_orr_loss_weight: 0.005
```

All flags follow the naming convention used by v50/v51 (`v52_*` prefix, snake_case, identity-init default).

---

## 5. Expected MPJPE Impact

| Scenario | Expected Δ MPJPE | Rationale |
|---|---|---|
| Smoke (RTX 4090, 2 epochs) | `-1.0` to `-2.5` mm | Outlier views are down-weighted before triangulation. |
| Full A800 | `-2.0` to `-4.0` mm | Physical-cue conditioning reduces uncertainty on floor/bone-consistent samples. |
| Sparse-view setting (`V<4`) | larger relative gain | Robust weighting becomes more valuable as each view has higher leverage. |

Baseline for comparison: v51 CDSVR smoke (~35 mm). We target `< 33` mm smoke with v52 enabled.

---

## 6. Risks

See `docs/swarm_iter26/reports/agent_outlier_robust_reliability_risks.md` for full risk register and mitigations. Top risks include:

1. **Gradient instability through M-estimator clipping.**
2. **Identity-init may collapse and never leave baseline.**
3. **Physical-cue branch can overfit to H36M floor plane.**
4. **Added latency from cross-view attention.**
5. **Interaction with v48 domain generalization causing domain-specific reliability collapse.**

---

## 7. 5-Step Implementation Plan

1. **Prototype the module** in `motionflow_mv/fusion/outlier_robust_reliability_v52.py` with `OutlierRobustReliabilityV52` class, Tukey bisquare helper, and physical-cue encoder.
2. **Wire it into `OmniMultiViewFusionV5`** (after v51 CDSVR, before triangulation/physical-space blocks) and add the config flags listed in §4.
3. **Add warm-start identity test**: assert that at init the module outputs equal its inputs (within tolerance) and that final sigmoid gates stay near 1.0.
4. **Run smoke test** on RTX 4090 with `configs/benchmark_v52_orr_smoke.yaml`; compare val_MPJPE against v51 CDSVR smoke baseline.
5. **If smoke `< 33` mm**, add an A800 queue entry in `scripts/launch_v33_a800_queue.py` and update `AGENTS.md` status tables; otherwise tune `v52_orr_alpha_init` and physical-cue loss weight.

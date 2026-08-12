# v53: Outlier-Robust Reliability (v53 ORR)

**Module key:** `outlier_robust_reliability_v53`  
**Depends on:** v25/v45 geometry fusion, v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, v52 Uncertainty-Weighted Triangulation  
**Tracking issue:** TBD (v53 ORR)

---

## 1. Motivation

v52 Uncertainty-Weighted Triangulation (UWT) already learns per-view/joint precision weights from feature statistics and reprojection residuals, but it still treats all residuals as unbounded squared errors. A single badly tracked, occluded, or temporarily mis-calibrated view can therefore dominate the triangulation because v52 has no explicit robust M-estimator and no way to *jointly* reason about view reliability across joints and physical invariants.

v53 Outlier-Robust Reliability (ORR) refines the v52 weights with a learned robust kernel. It fuses reprojection, epipolar, temporal, and physical-space residuals into a per-view/joint outlier score, then multiplies the v52 weights by a reliability factor that is identity at initialization. The result is a more resilient multi-view fusion stage before physical-space alignment, directly supporting the paper pipeline: *multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline*.

## 2. Architecture

### 2.1 Where it lives in `OmniMultiViewFusionV5`

```text
v25/v45 geometry fusion → v50 SEFH → v51 CDSVR → v52 UWT
                                                          ↓
                                              v53 ORR (this module)
                                                          ↓
                                                    weighted triangulation
                                                          ↓
                                            physical-space alignment (v28/v40)
```

v53 is called **immediately after v52 UWT**. It consumes the v52-predicted weights and the current 3-D estimate, computes robust residuals, predicts a multiplicative reliability factor, and returns refined weights for a second weighted-DLT pass or, equivalently, a gated correction around the v52 pose.

### 2.2 Internal blocks

| Block | Purpose | Output shape |
|---|---|---|
| `RobustResidualEncoder` | Compute robust residuals (reprojection, epipolar, temporal, physical). | `(B, T, V, J, 4)` |
| `ReliabilityRefiner` | Lightweight cross-view/joint attention to pool outlier evidence. | `(B, T, V, J, hidden)` |
| `RobustKernelHead` | Predict per-view/joint log-reliability `δ` and a learned robust scale `σ`. | `(B, T, V, J)` and scalar |

All final projection layers are zero-initialized; the multiplicative factor is therefore `exp(0) = 1` at startup, so `w_v53 = w_v52` until training begins.

### 2.3 Equations

Let

- `w_v52 ∈ [0,1]^{B×T×V×J}`: v52 triangulation weights.
- `e_reproj, e_epi ∈ ℝ^{B×T×V×J}`: reprojection and epipolar residuals from v50 SEFH.
- `e_temp ∈ ℝ^{B×T×J}`: temporal residual (joint velocity magnitude), broadcast to `(B,T,V,J)`.
- `e_phys ∈ ℝ^{B×T×J}`: physical-space residual (bone-length + floor-penetration energy), broadcast to `(B,T,V,J)`.

**Robust kernel (learned Charbonnier / adaptive scale):**

```
ρ_c(e; σ) = (e² + σ²)^{1/2} - σ
w_c(e)  = exp(-ρ_c(e; σ) / τ)
```

with `σ = Softplus(σ_log)` and `τ` a temperature. The learned `σ_log` is initialized so the kernel starts near L2.

**Reliability factor:**

```
feat   = concat([e_reproj, e_epi, e_temp, e_phys, log(1+e_reproj), log(1+e_epi)])  # (B,T,V,J,6)
h      = ReliabilityRefiner(feat)                                                   # (B,T,V,J,hidden)
δ      = RobustKernelHead(h)                                                         # (B,T,V,J)
γ      = exp(δ)                                                                    # identity at init (δ=0)
w_v53  = w_v52 * γ                                                                 # refined weights
```

**Refined triangulation / pose correction:**

```
X_v52 = DLT(points_2d, K, R, t, w_v52)
X_v53 = DLT(points_2d, K, R, t, w_v53)
X_out = X_v52 + g * (X_v53 - X_v52)                                                # g initialized to 0
```

`g` is a scalar or per-joint gate; zero initialization makes `X_out = X_v52` at startup.

### 2.4 Auxiliary loss

```
L_orr = (1/|V|) Σ_v w_v53 · ρ_c(e_reproj; σ) + λ_entropy · H(w_v53 / Σ_v w_v53)
```

The robust loss encourages low weights on views with large residuals; the entropy term prevents degenerate collapse to a single view.

## 3. Inputs and Outputs

### Inputs to `OutlierRobustReliabilityV53`

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_v52` | `(B, T, J, 3)` | 3-D pose from v52 UWT. |
| `w_v52` | `(B, T, V, J)` | v52 per-view/joint triangulation weights. |
| `points_2d` | `(B, T, V, J, 2)` | 2-D keypoints per view. |
| `K`, `R`, `t` | `(B, T, V, 3, 3)`, `(B, T, V, 3)` | Calibrated cameras. |
| `view_mask` | `(B, T, V)` | Valid-view mask. |
| `domain_id` | `(B,)` optional | Domain label (for per-domain robust scale). |

### Outputs

| Symbol | Shape | Description |
|---|---|---|
| `pred_3d_v53` | `(B, T, J, 3)` | Refined 3-D pose. |
| `w_v53` | `(B, T, V, J)` | Outlier-robust refined weights in `[v53_orr_min_weight, 1]`. |
| `orr_loss` | scalar | Auxiliary robust reliability loss. |
| `robust_scale` | scalar | Learned robust kernel scale (for logging). |

## 4. Config Flags

```yaml
use_v53_outlier_robust_reliability: false
v53_orr_hidden: 64
v53_orr_n_layers: 2
v53_orr_num_heads: 4
v53_orr_dropout: 0.1
v53_orr_kernel: "charbonnier"      # {"huber", "charbonnier", "tukey"}
v53_orr_identity_init: true
v53_orr_min_weight: 0.05
v53_orr_loss_weight: 0.01
v53_orr_use_reproj: true
v53_orr_use_epipolar: true
v53_orr_use_temporal: true
v53_orr_use_physical: true
v53_orr_temperature: 1.0
v53_orr_warmup_epochs: 0
v53_orr_residual_gate_init: -6.0    # logit for the pose-correction gate
```

## 5. Expected MPJPE Impact

| Scenario | Expected Δ MPJPE | Rationale |
|---|---|---|
| Identity check (loading v52 ckpt) | `< 0.1 mm` | Zero-init gates leave weights and pose unchanged. |
| Smoke (RTX 4090, 2 epochs) | `-0.5` to `-1.5 mm` vs v52 | Robust down-weighting of occasional outlier views. |
| Full A800 | `-1.0` to `-2.5 mm` vs v52 | Consistent handling of occlusion and calibration drift. |
| Sparse/variable views (`MPJPE@2/3`) | larger relative gain | Each view has higher leverage; robust re-weighting matters more. |

## 6. Risks

See `docs/swarm_iter27/reports/agent_outlier_robust_reliability_v53_risks.md` for the full risk register.

## 7. 5-Step Implementation Plan

1. **Implement `OutlierRobustReliabilityV53`** in `motionflow_mv/fusion/outlier_robust_reliability_v53.py` with `RobustResidualEncoder`, `ReliabilityRefiner`, and `RobustKernelHead`; enforce identity-at-init via zero-initialized final layers and `exp(0) = 1` reliability factors.
2. **Wire into `OmniMultiViewFusionV5`** immediately after the v52 UWT block: instantiate when `use_v53_outlier_robust_reliability=True`, call it with `pred_3d_gn_uwt`, `uwt_weights`, cameras, and 2-D points, and add `v53_orr_loss_weight * orr_loss` to `epi_loss` with a `v53_orr_warmup_epochs` guard.
3. **Add config flags** in the model `__init__` and create `configs/benchmark_v53_outlier_robust_reliability_smoke.yaml` mirroring the v52 smoke settings.
4. **Run smoke validation** on the local RTX 4090: confirm identity-at-init (delta `< 0.1 mm` versus v52), check for NaN/Inf/OOM, and compare epoch-1 `val_MPJPE` to the v52 baseline.
5. **Queue full A800 run** by adding an entry to `scripts/launch_v33_a800_queue.py` (e.g., `v53_outlier_robust_reliability_on_v52`) and update `AGENTS.md` once smoke results are in.

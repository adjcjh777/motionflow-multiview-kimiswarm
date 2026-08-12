# v52 — Reliability-Guided Pose Mixup (RGPM)

## Motivation

The current OmniMultiViewFusionV5 pipeline produces a single fused 3D pose from all available views. Modules such as v45 (adaptive geometry fusion), v50 (Self-Evolution Feedback Head), and v51 (Cross-Domain Sparse-View Reliability) improve the *weights* used in that single estimate, but they do not exploit the fact that, for a given multi-view frame, many plausible camera-subset pose hypotheses exist. Occlusions, calibration drift, and domain shift mean that the "best" pose is often better represented as a mixture of view-conditional hypotheses.

**v52 Reliability-Guided Pose Mixup** closes this gap. It generates a small set of alternative 3-D pose candidates by re-mixing the per-view geometry contributions according to the per-view reliability from v50/v51, then fuses those candidates with a lightweight per-joint transformer. The module is designed to be **identity-at-init**: if the candidate generator returns the original pose and the fusion residual is zero, the network behaves exactly like the baseline. This makes it safe to stack on top of v45+v50+v51 without destabilising an already-trained checkpoint.

This directly supports the paper story: multi-view video → human pose extraction → **multi-view fusion and calibration** → physical-space alignment → optimised motionflow pipeline.

## Architecture

### 1. Candidate generator

Assume the base geometry-fusion stage has produced, for each joint `j` and view `v`:

- `P_v ∈ R^(B×T×V×J×3)` — per-view 3-D point proposal (lifted/triangulated ray contribution),
- `w_v ∈ R^(B×T×V×J)` — geometry confidence/weight from v25/v45,
- `r_v ∈ R^(B×T×V×J)` — per-view reliability from v50/v51, bounded in `[0.05, 1]`,
- `σ_j ∈ R^(B×T×J)` — per-joint log-variance (uncertainty).

RGPM creates `M = v52_rgpm_num_candidates` candidate poses.

**Reliability-rescaled blending (primary, differentiable).** For each candidate `m`, sample mixing coefficients over views:

```
g_m ~ Gumbel(0, 1)                          # (B, T, V), reparameterised
logits_m = (log r_v + g_m) / τ              # τ = v52_rgpm_temperature
λ_m = softmax_v(logits_m)                   # (B, T, V), Σ_v λ_m(v) = 1
P_m = Σ_v λ_m(v) · w_v · P_v / Σ_v λ_m(v) · w_v    # (B, T, J, 3)
```

At `τ → 0` the candidate collapses to the most reliable view; at `τ → ∞` it becomes a uniform blend. The Gumbel noise is disabled at inference, so the candidate set is deterministic for evaluation.

**View-subset dropout candidate (auxiliary, no extra triangulation).** A binary mask `b_m ~ Bernoulli(π_v)` with `π_v = clamp(r_v, 0.1, 1.0)` zeroes out unreliable views before the same weighted blend. Because we blend existing `P_v` rather than re-triangulate, this is fully differentiable and cheap.

**Identity candidate.** Candidate 0 is always the original fused pose `P_0 = pred_3d` using the original v45 weights. This anchors the mixture to the baseline.

### 2. Fusion module

Candidates are embedded into per-joint tokens:

```
E_m(j) = Linear([P_m(j), σ_j, r̄_m(j)]) + joint_pos_embed(j)   # (B, T, M+1, hidden)
```

where `r̄_m(j)` is the mean reliability of candidate `m` for joint `j`, and the concatenation is over the feature dimension. A 2-layer transformer encoder with `v52_rgpm_num_heads` heads and `v52_rgpm_hidden` channels attends across candidates. Domain conditioning (optional) applies a learned FiLM layer using the v51 domain embedding.

The output is a per-joint residual:

```
Δ = W_o · TransformerEncoder(E)                         # (B, T, J, 3)
P_out = P_0 + γ · Δ                                      # γ = v52_rgpm_residual_gate_init, default 0.0
```

`W_o` is zero-initialised, and `γ` is initialised to 0.0, so the module starts as the identity map. During training, `γ` is learned as a scalar soft gate.

### 3. Auxiliary loss

A small consistency loss encourages the refined pose to agree with low-uncertainty candidates:

```
L_rgpm = w · E_m[ β_m · ||P_out - P_m||² ]
β_m ∝ exp(-reproj_err(P_m) / δ),  Σ_m β_m = 1
```

where `reproj_err(P_m)` is the average reprojection error of candidate `m`, and `w = v52_rgpm_loss_weight`. This loss is only applied during training and does not change the inference path.

## Inputs and outputs

**Inputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Base fused 3-D pose from v25/v45 |
| `per_view_proposals` | `(B, T, V, J, 3)` | Per-view 3-D point proposals `P_v` |
| `per_view_weights` | `(B, T, V, J)` | Geometry weights `w_v` from v25/v45 |
| `reliability` | `(B, T, V, J)` | v50/v51 per-view reliability `r_v` |
| `log_var` | `(B, T, J)` | v50/v51 per-joint log-variance `σ_j` |
| `points_2d` | `(B, T, V, J, 2)` | 2-D keypoint detections for reprojection error |
| `K, R, t` | `(B, V, 3, 3)`, `(B, V, 3, 3)`, `(B, V, 3)` | Calibrated cameras |
| `view_mask` | `(B, T, V)` | Valid-view mask |
| `domain_id` (optional) | `(B,)` | Domain label for v51-style conditioning |

**Outputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `refined_pred_3d` | `(B, T, J, 3)` | Reliability-mixup refined pose |
| `mixup_weights` | `(B, T, M)` | Candidate mixing weights for interpretability |
| `rgpm_loss` | `scalar` | Auxiliary consistency loss (training only) |

## Config flags

```yaml
use_v52_reliability_guided_pose_mixup: false
v52_rgpm_num_candidates: 4
v52_rgpm_dropout_prob: 0.3
v52_rgpm_hidden: 64
v52_rgpm_num_layers: 2
v52_rgpm_num_heads: 4
v52_rgpm_dropout: 0.1
v52_rgpm_residual_gate_init: 0.0
v52_rgpm_loss_weight: 0.01
v52_rgpm_temperature: 1.0
v52_rgpm_use_domain_conditioning: true
v52_rgpm_identity_at_init: true
```

## Expected MPJPE impact

- **H36M full-view benchmark:** modest ~0.5–1.0 mm improvement, because the base pose is already high quality.
- **Sparse/variable-view (MPJPE@2/MPJPE@3) and cross-domain 3DPW:** 2–5 mm improvement from robustness to missing/unreliable views.
- **Smoke target:** RTX 4090 smoke (50 samples, 1 epoch) should finish below 80 mm and without NaN/OOM.

## Risks

1. **Gumbel sampling noise can destabilise early training.** Mitigated by starting with high `τ` and annealing it, plus the identity candidate and zero-initialised output projection.
2. **Extra transformer capacity may overfit on small datasets.** Mitigated by dropout, identity-at-init, and freezing the base model for the first epoch.
3. **Reliability values from v50/v51 may be miscalibrated for some domains and steer mixup incorrectly.** Mitigated by clamping reliability to `[0.05, 1]` and making domain conditioning optional.
4. **Interaction with v51 CDSVR.** If CDSVR refines reliability differently per domain, the mixup generator must use the *refined* reliability, which adds a dependency flag. See the risk report for mitigations.

## 5-step implementation plan

1. **Module stub:** create `motionflow_mv/fusion/reliability_guided_pose_mixup_v52.py` with `ReliabilityGuidedPoseMixupV52` and a CPU smoke test verifying identity-at-init and output shapes.
2. **Wire into `OmniMultiViewFusionV5`:** add the flag block and call the module after v50/v51 in the forward, passing `pred_3d`, `per_view_proposals`, `reliability`, `log_var`, and optional `domain_id`.
3. **Loss and config:** add `v52_rgpm_loss_weight` to the training loss aggregator and add a smoke YAML under `configs/benchmark_v52_rgpm_smoke.yaml`.
4. **Local smoke:** run the smoke script on RTX 4090; verify val_MPJPE < 80 mm and that `refined_pred_3d ≈ pred_3d` at init.
5. **A800 full run and ablation:** queue a full A800 run with v45+v50+v51+v52, and run an ablation with v45+v50+v51 to isolate the v52 gain.

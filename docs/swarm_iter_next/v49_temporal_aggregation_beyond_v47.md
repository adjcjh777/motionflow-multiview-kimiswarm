# v49: Temporal Aggregation Beyond v47

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #167 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain)  

---

## 1. Problem statement

v47 adds a lightweight temporal transformer *on top of* the per-frame triangulated 3D pose. While this improves sparse-view coherence, it is intentionally limited:

1. **Single temporal scale.** v47 uses either a fixed local window or full-clip self-attention. It cannot simultaneously preserve fast motion (short window) and enforce long-range smoothness (full clip/cross-clip).
2. **Output-only refinement.** The temporal head refines the final 3D pose but never feeds information back into the sparse-view reliability/triangulation stage. Frames with few views stay noisy because the geometry-fusion stage sees each frame in isolation.
3. **No explicit uncertainty.** v47 does not expose a per-frame/per-joint temporal uncertainty that could be reused by the self-critique view-reliability loop (v37) or by v46's dynamic view selection.
4. **Memory grows with clip length.** Full-clip attention in v47 is impractical for very long clips or high-resolution skeletons.

v49 closes these gaps by adding a **multi-scale, uncertainty-aware temporal aggregator** that operates on the output pose and propagates temporal uncertainty back to the sparse-view triangulation step, strengthening the self-evolution feedback loop.

---

## 2. Proposed approach

### 2.1 Core idea

Replace/extend the single v47 temporal head with a **two-scale temporal module** plus a **temporal uncertainty head**:

- **Local branch:** Causal, small-window attention (`L_local = 5–7`) preserves sharp, fast motion and runs cheaply.
- **Global branch:** Dilated / lightweight full-clip attention (`L_global = clip_len`) enforces long-range smoothness without storing a dense `T×T` matrix (use strided attention or a Perceiver-style latent bottleneck, cf. v19/v31).
- **Uncertainty head:** Predicts per-frame, per-joint log-variance `σ_tj^2` from the two-scale features.
- **Feedback to sparse-view fusion:** The predicted temporal uncertainty rescales v37 self-critique reliability and/or v46 triangulation weights for the *same* frame, so the geometry stage learns to distrust temporally inconsistent joints.

At init the module is the identity map, so v47/v48 behavior is preserved during warm-up.

### 2.2 Architecture

```text
Input: per-frame triangulated pose P_t  (B, T, J, 3)
        |
        ▼
[ v49 Temporal Aggregation Module ]
        |
        ├── Local causal attention   (window = 5–7)
        ├── Global dilated/Perceiver attention (full clip)
        ├── Feature fusion (concat + linear)
        ├── Uncertainty head σ_t  (B, T, J)
        └── Residual refinement ΔP_t  (B, T, J, 3)
                |
                ▼
        Refined pose  P'_t = P_t + g · ΔP_t
                |
                ▼
        Feedback:  w_t = v46/v37 reliability * exp(-σ_t)
                  re-weight triangulation/confidence for frame t
```

### 2.3 Fit with v46-v48 and the overall pipeline

- **v46 Sparse-View Generalization:** v49 reuses the v46 view-dropout curriculum and reliability weights. The new uncertainty feedback directly modulates those weights, so sparse frames get stronger temporal smoothing and weaker per-frame confidence.
- **v47 Temporal Aggregation:** v49 is an optional upgrade. When `use_v49_temporal_aggregation=true`, the v47 path is bypassed; the YAML can still fall back to v47 by setting the flag to `false`.
- **v48 Domain Generalization:** The temporal module accepts per-domain conditioning (domain ID or domain embedding) so that studio and in-the-wild sequences can use different temporal smoothing strengths without a separate head per domain.
- **Overall multi-view video pipeline:** The pipeline is `per-view 2D → v25 geometry fusion → v46 sparse-view reliability → v49 temporal+uncertainty → v48 domain-invariant output`. v49 is the temporal reasoning layer; it does not replace the geometric foundation.

### 2.4 Self-evolution feedback loop

The v37 self-critique reliability estimator already predicts per-view reliability from reprojection residuals. v49 adds a **temporal self-consistency** signal:

1. After v49 produces `P'_t` and `σ_t`, compute the per-frame temporal residual `r_t = |P'_t - P_t|`.
2. Large `r_t` indicates that the raw per-frame triangulation is inconsistent with neighboring frames.
3. Use `σ_t` to down-weight the v37/v46 reliability scores for frame `t` in the *next* training iteration / self-evolution step:

```
reliability_feedback = reliability_v37 * exp(-α · σ_t)
```

where `α = v49_uncertainty_feedback_weight` (default 1.0).

This closes the loop: temporal aggregation critiques the per-frame geometry, the geometry stage adapts its view weights, and the next forward pass is temporally more consistent. It is inspired by the same self-improvement philosophy in v29 TTE and v30 AOSE, but implemented as a differentiable training-time feedback rather than an inference-only iterative loop.

---

## 3. Concrete code-level changes

### 3.1 New files

| File | Purpose |
|------|---------|
| `motionflow_mv/fusion/temporal_aggregation_v49.py` | `TemporalAggregationV49` module: local/global branches, uncertainty head, feedback helper. |
| `tests/test_temporal_aggregation_v49.py` | Unit tests for causality, variable clip length, and uncertainty shape. |
| `configs/benchmark_v49_temporal_beyond_v47_smoke.yaml` | Smoke config. |
| `scripts/run_v49_temporal_beyond_v47_smoke_local_4090.sh` | Smoke script. |

### 3.2 Modified files

| File | Change |
|------|--------|
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add `use_v49_temporal_aggregation` flag; instantiate v49 module; branch after v47 path. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags; pass `dataset_id` and `view_mask` to v49; add temporal uncertainty loss and feedback rescaling. |
| `experiments/eval_variable_views.py` | Report `temporal_jerk@k`, `uncertainty_mean@k`, and `MPJPE@k` for v47 vs v49. |

### 3.3 New training/evaluation flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_temporal_aggregation` | bool | `False` | Master switch. |
| `v49_temporal_d_model` | int | `64` | Hidden dimension for both temporal branches. |
| `v49_temporal_n_heads` | int | `4` | Attention heads. |
| `v49_temporal_local_window` | int | `7` | Local causal window size. |
| `v49_temporal_global_stride` | int | `2` | Stride for global dilated attention. |
| `v49_temporal_num_layers` | int | `2` | Shared encoder layers after feature fusion. |
| `v49_temporal_dropout` | float | `0.1` | Dropout. |
| `v49_temporal_loss_weight` | float | `0.01` | Weight of the temporal smoothness loss. |
| `v49_uncertainty_feedback_weight` | float | `1.0` | `α` in `reliability * exp(-α σ)`. `0.0` disables feedback. |
| `v49_use_perceiver_global` | bool | `True` | Use Perceiver-style latents for global branch; `False` uses strided self-attention. |
| `v49_perceiver_latents` | int | `32` | Number of latent tokens for the global branch. |
| `v49_use_domain_conditioning` | bool | `True` | Add domain embedding to temporal tokens (used with v48). |

### 3.4 Minimal YAML snippet

```yaml
model:
  # v46 / v47 / v48 base flags remain unchanged.
  use_v46_sparse_view_generalization: true
  use_v47_temporal_aggregation: false      # bypass v47
  use_v49_temporal_aggregation: true

  v49_temporal_d_model: 64
  v49_temporal_local_window: 7
  v49_temporal_global_stride: 2
  v49_temporal_num_layers: 2
  v49_temporal_dropout: 0.1
  v49_temporal_loss_weight: 0.01
  v49_uncertainty_feedback_weight: 1.0
  v49_use_perceiver_global: true
  v49_perceiver_latents: 32
```

---

## 4. Risks / failure modes

| Risk | Failure mode | Mitigation |
|------|--------------|------------|
| Multi-scale attention OOM | Global branch with dense `T×T` attention on long clips. | Use Perceiver latents or strided attention; default `clip_len=9` for smoke. |
| Over-smoothing fast motion | Local window too large or global branch too strong. | Keep local branch causal and window ≤7; initialise residual gate to 0.0. |
| Uncertainty feedback destabilizes training | Reliability scores collapse or explode. | Start with `v49_uncertainty_feedback_weight=0.0`, raise gradually; clip reliability to `[0.05, 0.95]`. |
| v46/v47 not yet merged | v49 cannot be wired. | Wait for #160/#162; implement v49 as a drop-in replacement so it slots in after v47 path. |
| No measurable gain over v47 | Temporal uncertainty does not correlate with error. | Add ablation: v49 without uncertainty head, v49 without global branch. |
| Domain shift | Studio vs in-the-wild temporal dynamics differ. | Enable `v49_use_domain_conditioning` and tune per-domain dropout in v48. |

---

## 5. Success metrics and recommended experiments

### 5.1 Metrics

- `MPJPE@k` for `k = 2, 3, 4, full` (same as v47).
- `temporal_jerk@k`: mean 3rd derivative of the refined trajectory (lower is smoother).
- `uncertainty_corr@k`: Pearson correlation between predicted `σ_t` and actual per-joint error; target `> 0.3`.
- `MPJPE@k Δ(v49 vs v47)`: relative improvement over v47 baseline.

### 5.2 Smoke test

| | |
|---|---|
| Hardware | RTX 4090 |
| Config | `configs/benchmark_v49_temporal_beyond_v47_smoke.yaml` |
| Recipe | 1 epoch, ~500 samples, `clip_len=9`, `d_model=64`, warm-start from best v48 checkpoint (or train end-to-end if unavailable). |
| Expected | `val_MPJPE < 75 mm`, no NaN/OOM, `temporal_jerk` lower than v47, uncertainty-error correlation `> 0.2`. |

### 5.3 Full experiment

| | |
|---|---|
| Hardware | A800-D |
| Base | Best v48 checkpoint (v47 enabled) or v48 from scratch if no checkpoint exists. |
| Config | `v49_temporal_beyond_v47_full.yaml` (`d_model=128`, `clip_len=13`, full WebBridge/H36M/MPI mixed manifest) |
| Goal | `MPJPE@2/3` improves ≥5% over v47; no regression `MPJPE@full`; `temporal_jerk` reduced ≥10%. |

### 5.4 Ablations

| Variant | Purpose |
|---------|---------|
| v49 local only | Disable global branch; isolate local-causal contribution. |
| v49 global only | Disable local branch; isolate long-range contribution. |
| v49 no uncertainty feedback | Keep uncertainty head but set `v49_uncertainty_feedback_weight=0.0`. |
| v49 no domain conditioning | Measure cross-domain gap with v48. |

---

## 6. Paper story fit

v49 supports the claim: *Our model not only handles sparse views and domain shift, but also reasons about temporal consistency at multiple scales and uses its own temporal uncertainty to self-correct the multi-view fusion step.* It turns the per-frame v46/v47 stack into a true temporal self-evolving system, which is a natural next step for the ICRA/CVPR 2027 multi-view video pipeline.

---

## 7. Relation to other variants

- **v47 Temporal Aggregation:** v49 is an optional successor; v47 remains available as a lighter fallback.
- **v49 real-time streaming (`docs/proposals/v49_realtime_streaming.md`):** That proposal targets causal online inference and view budgeting; this note targets richer offline temporal aggregation. They are complementary: streaming uses the causal local branch of v49 plus a GRU, while this design uses both local and global branches for highest accuracy.
- **v31/v19 temporal perceiver:** v49 reuses the Perceiver-latent idea for the global branch, but keeps the module small and focused on output-pose refinement.
- **v37 self-critique view reliability:** v49 feeds temporal uncertainty back into the v37 reliability path, closing the self-evolution loop.

---

## 8. Next steps

1. Wait for v47-temporal smoke results (#162) and v48-domain smoke results (#164).
2. Implement `TemporalAggregationV49` and unit tests.
3. Wire v49 flags into `OmniMultiViewFusionV5` and the trainer.
4. Create smoke config/script and run on RTX 4090.
5. Compare `MPJPE@k`, `temporal_jerk`, and uncertainty-error correlation against the v47 baseline.
6. If smoke meets targets, queue a full A800 run starting from the best v48 checkpoint.

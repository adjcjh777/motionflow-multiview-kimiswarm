# v49: Self-Evolution with Uncertainty / Reliability / Reprojection Feedback

**Status:** Proposal / design ready  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #166 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain)

---

## 1. Problem statement

By v48 the pipeline is robust to sparse views, temporal noise, and domain shift, but it still triangulates once and stops.  The per-view confidences produced by v45-AGF and the v46 reliability head are trained *offline*; they do not react to the geometric self-consistency of the current prediction.  This leaves two gaps:

1. **Residual information is discarded.**  After triangulation we know, for every view and joint, how well the predicted 3D point reprojects onto the original 2D detections.  Large residuals flag bad views/occlusions, yet the model does not feed them back to correct the next estimate.
2. **Uncertainty is under-used.**  v36/v43 per-node uncertainty gates and v37 self-critique reliability scores are learned from static features.  They are never updated by the actual triangulation error, so they cannot self-correct when a view that looked reliable is in fact wrong.

The TTE-style test-time loop (v27) that tried to close this loop was unstable and produced ~90 mm failures, so we need a **training-time, gradient-safe self-evolution feedback** mechanism that learns to reduce uncertainty from reprojection and temporal residuals, rather than iterating the frozen model at inference.

---

## 2. Proposed approach and how it fits with v46-v48 / the overall pipeline

v49 adds a single lightweight `SelfEvolutionUncertaintyFeedbackV49` head on top of the v48 stack.  It re-uses existing v45/v46 reliability scores and v36/v43 uncertainty gates, but makes them **iteratively refinable** from three feedback signals:

| Feedback signal | What it captures | How v49 uses it |
|-----------------|------------------|-----------------|
| Reprojection residual | Per-(view,joint) consistency of the triangulated 3D pose with 2D detections | Target reliability/uncertainty update: high residual → lower reliability, higher uncertainty |
| Temporal consistency residual | Disagreement with the v47 temporal-smoothed trajectory | Penalise unreliable views that cause jumps |
| Cross-view epipolar residual | Ray-pair reprojection error from other views | Additional robustness cue for few-view cases |

### Pipeline diagram

```text
2D keypoints + cameras
       |
       v
[v25 Multi-View Geometry Fusion]
       |
       v
[v45/v46 reliability weights  r_vj  and v36/v43 uncertainty  u_vj]
       |
       v
[v48 Domain-invariant sparse-view refinement]  ->  P_t
       |
       v
[SelfEvolutionUncertaintyFeedbackV49]
       |
       |-- Reprojection residual  e_reproj(P_t)
       |-- Temporal residual      e_temp(P_t, P_{t-1}, P_{t+1})
       |-- Epipolar residual      e_epi(P_t)
       |
       v
[updated reliability  r'_vj = g(r_vj, u_vj, e_reproj, e_temp, e_epi)]
       |
       v
[second-pass weighted triangulation / residual refinement]  ->  P'_t
```

### Fit with v46-v48

- **v46 Sparse-View Generalization:** v49 consumes the v46 reliability head and lets it evolve from feedback; the view-dropout augmentation is unchanged.
- **v47 Temporal Aggregation:** v47 provides the smoothed trajectory used to compute the temporal residual.  v49 can be inserted after v47 or alongside it.
- **v48 Domain Generalization:** domain-conditional FiLM features from v48 are kept; the new feedback head is domain-agnostic by default (no per-domain parameters).
- **Overall multi-view pipeline:** the v48 model remains the backbone.  v49 is a post-triangulation self-correction that closes the loop between 2D evidence, 3D prediction, and per-view uncertainty.

### Key design choice: train-time only, gradient-safe

Unlike the broken v27 TTE module, v49 does **not** iterate a frozen triangulation head at inference.  Instead, during training it predicts a *residual refinement* and an updated uncertainty map, both supervised by the final pose loss and auxiliary residual losses.  At inference the module is a single forward pass, so it cannot diverge or add latency.

---

## 3. Concrete code-level changes

### New module

`motionflow_mv/fusion/self_evolution_uncertainty_feedback_v49.py`

```python
class SelfEvolutionUncertaintyFeedbackV49(nn.Module):
    def __init__(
        self,
        n_joints: int = 17,
        n_views: int = 8,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_temporal_feedback: bool = True,
        use_epipolar_feedback: bool = True,
        residual_gate_init: float = 0.0,
    ):
        ...

    def forward(
        self,
        pose_3d: torch.Tensor,              # (B, T, J, 3)
        points_2d: torch.Tensor,            # (B, T, V, J, 2)
        K: torch.Tensor,                    # (B, T, V, 3, 3)
        R: torch.Tensor,                    # (B, T, V, 3, 3)
        t: torch.Tensor,                    # (B, T, V, 3)
        reliability: torch.Tensor,          # (B, T, V, J)
        uncertainty: torch.Tensor,          # (B, T, V, J)
        view_mask: torch.Tensor,            # (B, T, V)
        temporal_pose: torch.Tensor | None = None,  # (B, T, J, 3) from v47
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            refined_pose: (B, T, J, 3)
            updated_reliability: (B, T, V, J) in [0, 1]
            updated_uncertainty: (B, T, V, J) positive
        """
```

### Files to touch

| File | Change |
|------|--------|
| `motionflow_mv/fusion/self_evolution_uncertainty_feedback_v49.py` | New module. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add v49 flags; call the module after v48/v47 output; feed it reprojection/epipolar/temporal residuals. |
| `motionflow_mv/losses/reprojection_loss.py` (or create) | Helper for per-view/joint reprojection and epipolar residuals. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags; add auxiliary self-evolution losses; log updated reliability/uncertainty histograms. |
| `experiments/eval_variable_views.py` | Report per-view/joint mean uncertainty and mean reprojection residual alongside MPJPE@k. |
| `tests/test_self_evolution_uncertainty_feedback_v49.py` | Unit tests for shape, mask handling, identity at init, and residual computation. |
| `configs/benchmark_v49_self_evolution_uncertainty_feedback_smoke.yaml` | Smoke config. |
| `scripts/run_v49_seuf_smoke_local_4090.sh` | Smoke script. |

### New training / model flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_self_evolution_uncertainty_feedback` | bool | `False` | Master switch. |
| `v49_seuf_hidden` | int | `64` | MLP hidden dim for feedback head. |
| `v49_seuf_num_layers` | int | `2` | Number of MLP layers. |
| `v49_seuf_dropout` | float | `0.1` | Dropout in the feedback MLP. |
| `v49_seuf_use_temporal_feedback` | bool | `True` | Include temporal residual (requires v47). |
| `v49_seuf_use_epipolar_feedback` | bool | `True` | Include epipolar residual. |
| `v49_seuf_reproj_loss_weight` | float | `0.01` | Weight of the auxiliary reprojection-residual loss. |
| `v49_seuf_temporal_loss_weight` | float | `0.005` | Weight of the temporal consistency loss. |
| `v49_seuf_uncertainty_loss_weight` | float | `0.01` | Weight of the uncertainty-target loss. |

### Losses

Three auxiliary losses supervise the feedback head:

1. **Reprojection-target loss**
   ```
   target_reliability = sigmoid(-reproj_err * 10)
   L_reli = MSE(updated_reliability, target_reliability)
   ```
2. **Uncertainty-target loss**
   ```
   target_uncertainty = log(1 + reproj_err)
   L_unc = MSE(updated_uncertainty, target_uncertainty)
   ```
3. **Temporal consistency loss** (when `v49_seuf_use_temporal_feedback` is true)
   ```
   L_temp = mean(|refined_pose_t - temporal_pose_t|) * confidence_t
   ```

Total auxiliary loss:
```
L_seuf = v49_seuf_reproj_loss_weight * L_reli
       + v49_seuf_uncertainty_loss_weight * L_unc
       + v49_seuf_temporal_loss_weight * L_temp
```

The final training loss is `L_pose + L_seuf`.

---

## 4. Risks / failure modes

| Risk | How it manifests | Mitigation |
|------|------------------|------------|
| **Reliability/uncertainty collapse** | Updated scores become constants; no view is down-weighted. | Strong target losses tied to actual residuals; small hidden layers to prevent memorisation. |
| **v27-like TTE re-introduced** | Attempting to iterate at inference causes divergence. | Keep v49 as a single residual refinement during training; inference is one forward pass. |
| **Residual gate explodes** | Refined pose drifts far from v48 baseline. | Initialise residual gate to 0; apply `tanh`/clamp on pose residual; use small learning rate. |
| **Epipolar residual is noisy at 2 views** | Degenerate geometry amplifies residual signal. | Disable `v49_seuf_use_epipolar_feedback` when `V < 3`; fall back to reprojection-only feedback. |
| **Slowdown from extra residuals** | Computing reprojection/epipolar residuals per step increases training time. | Cache residuals inside `omniview_fusion_v5.py` forward; only compute when v49 is enabled. |

---

## 5. Success metrics and recommended smoke / full experiment

### Metrics

- `val_MPJPE@k` for `k ∈ {2, 3, 4, full}` (reuse `experiments/eval_variable_views.py`).
- `mean_updated_uncertainty_bad_view`: average uncertainty of the worst 10 % of views by reprojection error.
- `mean_updated_reliability_good_view`: average reliability of the best 50 % of views.
- Correlation between updated uncertainty and actual reprojection error (target Spearman > 0.5).

### Smoke experiment

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Smoke | RTX 4090 | `configs/benchmark_v49_self_evolution_uncertainty_feedback_smoke.yaml` | val_MPJPE finite; no NaN/OOM; correlation uncertainty↔reproj > 0.3 |

```bash
bash scripts/run_v49_seuf_smoke_local_4090.sh
```

Typical smoke config: `d=64`, `train_samples=500`, `clip_len=9`, all v49 losses enabled with weights above.

### Full experiment

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Full | A800-D | `v49_seuf_all_train` manifest, warm-start from best v48 checkpoint | Reduce val_MPJPE by ≥2 % over v48 at sparse views; no regression at full views |

Recipe:
1. Warm-start from the best v48-domain checkpoint.
2. Freeze v25/v45/v46/v47/v48 weights for 1 epoch; train only the v49 feedback head.
3. Unfreeze and fine-tune end-to-end with the v48 mixed manifest.
4. Validate on H36M/MPI/AIST val, 3DPW pseudo val, and 3DPW actual val.

### Success criteria

1. Smoke test passes with no NaN/OOM and finite per-domain val_MPJPE.
2. v49 full-run `val_MPJPE@k` is within 1 mm of v48 at full views (no regression).
3. At `k ≤ 3`, v49 improves over v48 by ≥2 % MPJPE.
4. Updated reliability/uncertainty scores correlate with actual reprojection errors (Spearman > 0.5).
5. A800 full run completes ≥1 epoch.

---

## 6. The self-evolution feedback loop

The central idea of v49 is a **closed loop** between prediction, observation, and uncertainty:

```text
predict P_t  ->  measure residuals  ->  update r_vj and u_vj
      ^                                              |
      |                                              v
      |                                  refine P_t using updated r_vj, u_vj
      |______________________________________________|
```

- **Prediction:** v48 produces the initial 3D pose `P_t` and the v45/v46/v48 reliability/uncertainty maps.
- **Observation:** v49 computes reprojection, temporal, and epipolar residuals for `P_t`.
- **Self-critique:** the feedback MLP maps `(residuals, old reliability, old uncertainty)` to updated reliability and uncertainty.
- **Refinement:** updated scores re-weight a second-pass triangulation, producing `P'_t`.
- **Learning:** the final pose loss and the auxiliary residual losses train the feedback MLP to produce *better* uncertainty estimates on the next sample.

This is the multi-view analogue of Qwen-style self-improvement: the model learns to critique its own multi-view predictions and down-weight inconsistent evidence, but it does so **inside the training graph** so the loop is stable and differentiable.

### Relation to existing variants

- **v27 TTE:** v27 iterated the frozen triangulation head at inference and broke; v49 learns a one-step refinement during training and is inference-safe.
- **v37 SCVR:** v37 learned per-view reliability from static tokens; v49 *updates* that reliability from geometric residuals.
- **v39 RCAGR:** v39 coupled reliability to the graph-refinement gate; v49 generalises that coupling to reprojection/temporal/epipolar feedback.
- **v43 adaptive per-node residual:** v43 scaled residuals by static uncertainty; v49 makes uncertainty itself evolve from feedback.
- **v45/v46:** v49 consumes the per-view/per-joint weights from v45-AGF/v46-SVG and refines them.

### Next steps

1. Wait for v48-domain smoke results (#164).
2. Implement `SelfEvolutionUncertaintyFeedbackV49` and unit tests.
3. Wire v49 flags into `OmniMultiViewFusionV5` and the trainer.
4. Run smoke on RTX 4090 and verify uncertainty-reprojection correlation.
5. Queue full A800 run warm-started from the best v48 checkpoint.

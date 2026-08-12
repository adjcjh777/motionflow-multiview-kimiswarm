# v49: Cross-View Consistency Losses

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #166  
**Depends on:** v45-AGF (#154), v46-SVG (#160), v48-domain (#164)  

---

## 1. Problem statement

v45 Adaptive Geometry Fusion and v46 Sparse-View Generalization improve triangulation by learning per-view reliability, but the training objective still treats multi-view consistency indirectly: a single global reprojection loss is applied *after* the final 3-D estimate, and the existing `CrossViewJointContrastiveLoss` (`motionflow_mv/losses/crossview_pose_contrast.py`) is not wired into the current `OmniMultiViewFusionV5` pipeline. As a result:

1. **No explicit cross-view agreement signal** before or during triangulation; noisy/occluded views can still bias the DLT solve before the downstream reprojection loss can correct them.
2. **v46 reliability is only used as a weight**, not supervised by a consistency objective, so the reliability head may drift when views are dropped or domains change.
3. **The self-evolution loop is incomplete.** v37 self-critique and v45 adaptive weights already consume reprojection residuals, but those residuals are computed *after* the pose is final. A richer set of cross-view consistency losses can provide earlier, per-view feedback.

v49 therefore adds a **small suite of cross-view consistency losses** that operate on per-view 2-D observations, per-view features, and the fused 3-D pose, gated by v45/v46 reliability.

---

## 2. Proposed approach and fit with v46–v48

Add a single new loss module, `CrossViewConsistencyLossesV49`, that exposes four optional terms. Each term is gated by the existing `view_mask` and, where available, the v45/v46 reliability weights:

| Term | What it enforces | Inputs |
|------|------------------|--------|
| **Reliability-weighted reprojection** | The final 3-D pose projects back into each view with low error; unreliable/dropped views are down-weighted. | `pred_3d`, `points_2d`, `K`, `R`, `t`, `reliability`, `view_mask` |
| **Epipolar / ray consistency** | Each pair of 2-D observations of the same joint is geometrically consistent with the calibrated camera pair. | `points_2d`, `K`, `R`, `t`, `view_mask` |
| **Triangulation agreement** | Pair-wise (or triplet-wise) triangulations agree with the full-view fused pose, improving few-view robustness. | `pred_3d`, `points_2d`, `K`, `R`, `t`, `view_mask` |
| **Cross-view joint contrastive** | Same-joint features across views are pulled together, different joints are pushed apart. | per-view joint features, `view_mask` |

### How it fits the pipeline

```text
per-view 2-D keypoints + cameras
        |
        v
[v25 Multi-View Geometry Fusion]
        |
        v
[v45 Adaptive Geometry Fusion]  ----> per-(view,joint) reliability r_vj
        |
        v
[v46 Sparse-View Generalization]  ----> masked/dropped view handling
        |
        v
pred_3d  +  per-view features
        |
        v
[v49 Cross-View Consistency Losses]
        |-- reliability-weighted reprojection
        |-- epipolar/ray consistency
        |-- triangulation agreement
        |-- cross-view joint contrastive
        |
[v47 Temporal Aggregation / v48 Domain Generalization]
```

- **v46 SVG:** provides `view_mask` and reliability scores that weight every v49 term, so dropped views do not contribute to consistency losses.
- **v47 Temporal:** can reuse the per-view residuals produced by v49 as an additional temporal smoothness target (e.g., penalise frame-to-frame changes in reprojection error).
- **v48 Domain:** the losses are computed on the surviving per-domain views and encourage domain-invariant cross-view features, complementing the v48 GRL/FiLM objective.

### Self-evolution feedback loop

v49 closes the action→feedback→retry loop that underpins v36–v45:

1. The model produces per-view 2-D observations and a fused 3-D pose.
2. v49 computes per-view reprojection residuals, epipolar violations, and triangulation agreement.
3. These residuals become **supervision targets** for the v37 self-critique reliability head (`target = sigmoid(-reproj_err * 10)`).
4. v45 adaptive weights are updated from the same residuals.
5. In the next forward pass, the improved reliability weights and gated losses produce a more consistent pose.

---

## 3. Concrete code-level changes

### New module

`motionflow_mv/losses/cross_view_consistency_v49.py`

```python
class CrossViewConsistencyLossesV49(nn.Module):
    def __init__(
        self,
        reproj_weight: float = 0.0,
        epipolar_weight: float = 0.0,
        triang_agree_weight: float = 0.0,
        contrastive_weight: float = 0.0,
        temperature: float = 0.07,
        use_reliability_gate: bool = True,
        warmup_epochs: int = 1,
    ):
        ...

    def forward(
        self,
        pred_3d: torch.Tensor,              # (B, T, J, 3)
        points_2d: torch.Tensor,            # (B, T, V, J, 2)
        K: torch.Tensor,                    # (B, T, V, 3, 3)
        R: torch.Tensor,                    # (B, T, V, 3, 3)
        t: torch.Tensor,                    # (B, T, V, 3)
        per_view_features: torch.Tensor | None = None,  # (B, T, V, J, d)
        reliability: torch.Tensor | None = None,        # (B, T, V, J)
        view_mask: torch.Tensor | None = None,        # (B, T, V)
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        ...
```

### Files touched

| File | Change |
|------|--------|
| `motionflow_mv/losses/cross_view_consistency_v49.py` | New module with the four loss terms. |
| `motionflow_mv/losses/__init__.py` | Export `cross_view_consistency_loss_v49`. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add v49 flags and, when enabled, return per-view joint features needed by the contrastive term. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags, call the v49 module in `compute_loss`, log per-term metrics. |
| `configs/benchmark_v49_cross_view_consistency_smoke.yaml` | Smoke config. |
| `scripts/run_v49_cross_view_consistency_smoke_local_4090.sh` | Smoke script. |
| `tests/test_cross_view_consistency_v49.py` | Unit tests for each term and view-mask handling. |

### New training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_cross_view_consistency` | bool | `False` | Master switch. |
| `v49_cvc_reproj_weight` | float | `0.0` | Weight for reliability-weighted reprojection loss. |
| `v49_cvc_epipolar_weight` | float | `0.0` | Weight for epipolar/ray consistency loss. |
| `v49_cvc_triang_agree_weight` | float | `0.0` | Weight for pair-wise triangulation agreement loss. |
| `v49_cvc_contrastive_weight` | float | `0.0` | Weight for cross-view joint contrastive loss. |
| `v49_cvc_temperature` | float | `0.07` | Temperature for the contrastive loss. |
| `v49_cvc_use_reliability_gate` | bool | `True` | Multiply consistency terms by v45/v46 reliability. |
| `v49_cvc_warmup_epochs` | int | `1` | Linear ramp-up of all v49 weights. |

### Wiring sketch

Inside `experiments/train_omniview_fusion_v5_webbridge_multi.py`, after the existing supervised loss:

```python
if args.use_v49_cross_view_consistency:
    cvc_dict = cross_view_consistency_loss_v49(
        pred_3d,
        x[..., :2],
        K_aug,
        R_aug,
        t_aug,
        per_view_features=features,   # returned from OmniMultiViewFusionV5 when enabled
        reliability=reliability,      # from v45/v46
        view_mask=view_mask,
        epoch=epoch,
    )
    # warmup
    cvc_total = cvc_dict["total"] * min(1.0, (epoch + 1) / args.v49_cvc_warmup_epochs)
    loss = loss + cvc_total
    metrics.update({k: v.item() for k, v in cvc_dict.items()})
```

---

## 4. Risks / failure modes

| Risk | Mitigation |
|------|------------|
| **Loss dominates early training.** | Linear warmup (`v49_cvc_warmup_epochs`) and identity-at-init for any new projection heads; start with all weights at `0` and enable one term at a time. |
| **Pair-wise triangulation is expensive.** | Cache ray directions; only sample a fixed number of view pairs per forward, or limit to triplets when `V > 8`. |
| **Contrastive positives are scarce under heavy dropout.** | Use `view_mask` to exclude invalid anchors; fall back to joint-only negatives when `k < 2`. |
| **Double-counting with v25 geometry losses.** | v49 epipolar term overlaps with the existing `epi_loss`; start with `v49_cvc_epipolar_weight=0` and add only if ablations show gain. |
| **v46 reliability not yet calibrated.** | Set `v49_cvc_use_reliability_gate=False` for the first smoke to isolate the raw consistency signal. |
| **Gradient instability through triangulation.** | Stop gradients into `points_2d` for the triangulation-agreement term; only back-propagate through the fused pose. |

---

## 5. Success metrics and recommended experiments

### Primary metrics

| Metric | How to measure |
|--------|----------------|
| `val_MPJPE` | Standard mixed H36M/MPI validation. |
| `reproj_error@k` | Mean per-view reprojection error at view count `k`. |
| `triang_agreement_mm` | Mean disagreement between pair-wise triangulations and the fused pose. |
| `contrastive_loss` | Magnitude of the contrastive auxiliary loss. |

### Experiments

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| **Smoke** | RTX 4090 | `configs/benchmark_v49_cross_view_consistency_smoke.yaml` | val_MPJPE < 80 mm; no NaN/OOM; cross-view reprojection residual lower than v45/v46 baseline. |
| **Full** | A800-D | v48-domain base + v49 flags | ≥5% MPJPE improvement at `k = 2, 3`; no regression at full views. |
| **Ablation** | RTX 4090 | each v49 term enabled in isolation | Identify which terms are essential; avoid double-counting with v25. |

### Minimal smoke command

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --config configs/benchmark_v49_cross_view_consistency_smoke.yaml \
  --use_v49_cross_view_consistency \
  --v49_cvc_reproj_weight 0.1 \
  --v49_cvc_epipolar_weight 0.05 \
  --v49_cvc_triang_agree_weight 0.05 \
  --v49_cvc_contrastive_weight 0.05 \
  --v49_cvc_use_reliability_gate \
  --v49_cvc_warmup_epochs 1
```

### Expected outcome

- Smoke: the v49-augmented run matches or improves on the v45/v46 smoke baseline, with a measurable reduction in per-view reprojection error.
- Full A800: the sparse-view regime (`MPJPE@2`, `MPJPE@3`) improves by ≥5% relative to v48, while `MPJPE@full` stays within 1 mm.

---

## 6. Self-evolution feedback loop

v49 is the **multi-view consistency verifier** in the self-evolution stack:

1. **Produce:** `OmniMultiViewFusionV5` emits per-view features and a fused 3-D pose.
2. **Verify:** `CrossViewConsistencyLossesV49` measures per-view reprojection residuals, epipolar violations, and triangulation agreement.
3. **Diagnose:** Residuals are converted into per-(view,joint) reliability targets for v37 and into adaptive weights for v45.
4. **Retry:** In subsequent iterations the improved reliability weights and gated losses produce a more geometrically consistent pose.

Because v49 losses are differentiable, the model learns to predict higher reliability for views that satisfy cross-view consistency and lower reliability for outlier/corrupted views, closing the loop without manual thresholds.

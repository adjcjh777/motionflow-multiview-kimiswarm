# v80 Sparse-View Robust Multi-View Pose (SV-RMVP)

**Status:** Proposal  
**Labels:** `experiment`, `P1-next`  
**Depends on:** v46 Sparse-View Generalization, v52 Uncertainty-Weighted Triangulation, v57 Domain-Conditional Physical-Space Calibration  

---

## 1. Motivation: why sparse-view robustness is the right differentiator

The legacy H36M leaderboard was measuring how closely a network reproduced the DLT layer, because the H36M labels were circular (`joints_3d == DLT(points_2d, cameras)`). With the pivot to **MPI-INF-3DHP** and **Shelf/Campus**—datasets that have true 3D GT—the paper story must shift from chasing an illusory MPJPE record to solving a real deployment problem: **cameras are missing, occluded, or ad-hoc in practice**.

Sparse-view robustness is the natural differentiator because:

1. **Real rigs are not 14-view H36M.** Most in-the-wild or clinical captures use 2–6 cameras, and one or more views are often occluded.
2. **Triangulation quality is view-limited.** At 2–4 views, the standard DLT/geometry-fusion baseline degrades quickly; a learned view-reliability mechanism can compensate.
3. **Existing assets already point this way.** v46 (view dropout + per-view reliability), v52 (uncertainty-weighted triangulation), and v57 (domain-conditional physical-space calibration) are proven building blocks. v80 ties them together around one new idea: **learned pre-triangulation view reliability with a lightweight test-time view-subset search**.

---

## 2. Proposed architecture: v80 Sparse-View Robust Multi-View Pose

v80 inserts a small **View-Reliability Before Triangulation (VRBT)** module upstream of v52, then lets v52/v57 refine the result. Optionally, at inference time it runs a cheap **Test-Time View-Subset Search (TTVSS)** that picks the highest-reliability subset of available views for triangulation.

```text
Input: (B, T, V, J, C) feature tokens + (B, T, V, J, 2) 2D keypoints + cameras
        |
        ▼
[ v25 Multi-View Geometry Fusion ]
        |
        ▼
[ v80 VRBT head ]
        |
        ├── Per-view geometry-bias features (reproj residual, epipolar consistency,
        │    feature-token mean/std across views).
        ├── Domain-conditional reliability MLP (conditions on v57-style domain embedding).
        └── Outputs per-view reliability r_v ∈ (0,1) and a subset quality score q_k.
        |
        ▼
[ v52 Uncertainty-Weighted Triangulation ]
        |   Uses r_v as the prior weights_prior; v52 predicts residual precision.
        ▼
[ v57 Domain-Conditional Physical-Space Calibration ]
        |   Domain-specific floor / bone / residual calibration.
        ▼
Output: refined 3D pose P'_t
```

### New idea: VRBT + TTVSS

- **VRBT** predicts a reliability weight for each *available* view *before* any triangulation, using only the 2D keypoints, cameras, and fused feature tokens. It is trained with an self-supervised objective: the predicted reliability should correlate with the reprojection error of the eventual v57-corrected pose.
- **TTVSS** (inference-only) enumerates candidate subsets sizes `k ∈ {2, 3, 4, full}` and selects the subset whose sum of VRBT reliability scores, normalized by subset size, is highest. The selected subset is fed into v52/v57. This makes the model explicitly robust to the worst view being noisy, without retraining.

The whole module is identity at initialization (final layers zero-initialized), so v80 starts as the v52/v57 baseline and learns to improve only where the data supports it.

---

## 3. Input/output interface and pipeline fit

### Module API

```python
class SparseViewReliabilityV80(nn.Module):
    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        n_joints: int = 17,
        hidden: int = 64,
        num_domains: int = 8,
        weight_type: str = "per_view_joint",
        use_test_time_subset_search: bool = True,
        subset_sizes: Tuple[int, ...] = (2, 3, 4, -1),
    ):
        ...

    def forward(
        self,
        features: Tensor,      # (B, T, V, J, d)
        points_2d: Tensor,     # (B, T, V, J, 2)
        pred_3d_init: Tensor,  # (B, T, J, 3)
        K: Tensor,             # (B, T, V, 3, 3)
        R: Tensor,             # (B, T, V, 3, 3)
        t: Tensor,             # (B, T, V, 3)
        view_mask: Tensor,     # (B, T, V)
        domain_id: Tensor,     # (B,)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            reliability: (B, T, V, J) prior weights for v52.
            subset_mask: (B, T, V) optional boolean mask for TTVSS.
        """
```

### Integration in `OmniMultiViewFusionV5`

Add the module call inside the existing v52 branch in `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
if self.use_v52_uncertainty_weighted_triangulation:
    if self.use_v80_sparse_view_reliability:
        v80_weights, v80_subset_mask = self.v80_sparse_view_reliability(
            features, points_2d, pred_3d, K, R, t, view_mask, domain_id
        )
        # Combine with existing confidences and masks.
        weights_prior = confidences * v80_weights
        if v80_subset_mask is not None:
            view_mask = view_mask & v80_subset_mask
    else:
        weights_prior = confidences

    pred_3d, uwt_loss, weights, log_prec = self.v52_uwt(
        features, points_2d, K, R, t, pred_3d,
        view_mask=view_mask,
        domain_id=domain_id,
        weights_prior=weights_prior,
    )
```

v57 remains downstream of v52 and receives the same inputs plus the v80 weights for floor/bone conditioning.

---

## 4. Expected experimental gains (qualitative)

The new target datasets are MPI-INF-3DHP and Shelf/Campus, evaluated with true 3D GT via `experiments/eval_variable_views.py`.

| Metric | Expected behavior |
|--------|-------------------|
| `MPJPE@2` / `MPJPE@3` | Improves over v52/v57 baseline because VRBT down-weights outlier/noisy views before triangulation. |
| `MPJPE@4` / `MPJPE@full` | Matches v57 baseline (identity init avoids regression). |
| 4-view vs. 14-view gap | Shrinks; TTVSS lets the model use the best subset even when only 2–4 views are available. |
| Cross-domain | v80 reliability MLP is domain-conditioned, so the same checkpoint adapts to MPI vs. Shelf/Campus. |

Target smoke-test criterion: on MPI-INF-3DHP val, `MPJPE@2` drops by ≥10% relative to v57 without v80.

---

## 5. Risks and ablations

| Risk | Mitigation / ablation |
|------|-----------------------|
| VRBT overlaps with v52 per-view precision prediction. | Treat VRBT as a *prior* and v52 as *posterior* precision; ablate removing one or the other. |
| TTVSS is too expensive at high view counts. | Limit subset sizes to `k ≤ 4` and cache scores; make it optional (`use_v80_ttvss`). |
| Domain-conditioning overfits to small datasets. | Use shared domain embedding + FiLM; freeze domain embedding in first epoch. |
| Sparse-view training destabilizes full-view accuracy. | Identity initialization + curriculum (ramp view dropout). |

### Ablations to run

1. **v80 without VRBT** (only TTVSS over uniform weights).
2. **v80 without TTVSS** (VRBT prior only).
3. **v80 without domain conditioning** (single-domain reliability MLP).
4. **v80 with fixed `k=4`** subset vs. learned subset.

---

## 6. Implementation sketch

### New files

- `motionflow_mv/fusion/sparse_view_reliability_v80.py` — `SparseViewReliabilityV80` module.
- `motionflow_mv/eval/test_time_view_subset_search.py` — helper to enumerate and score subsets.
- `configs/benchmark_v80_sparse_view_robust_smoke.yaml` — smoke config.
- `scripts/run_v80_sparse_view_robust_smoke_local_4090.sh` — smoke script.

### Files to touch

| File | Change |
|------|--------|
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add `use_v80_sparse_view_reliability`, instantiate `SparseViewReliabilityV80`, feed `weights_prior` to v52, apply `subset_mask`. |
| `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py` | Already accepts `weights_prior`; no change unless debug. |
| `motionflow_mv/fusion/domain_conditional_physical_calibration_v57.py` | Accept optional v80 weights for floor/bone robustness (default behavior unchanged). |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags, pass `domain_id` and `view_mask` to v80, log `MPJPE@k`. |
| `experiments/eval_variable_views.py` | Add `--use_v80_ttvss` flag; report `MPJPE@k` for `k=2,3,4,full`. |

### New flags

```yaml
model:
  use_v80_sparse_view_reliability: true
  v80_svr_hidden: 64
  v80_svr_num_layers: 2
  v80_svr_weight_type: "per_view_joint"
  v80_svr_use_domain_conditioning: true
  v80_svr_identity_init: true
  v80_svr_min_weight: 0.05
  v80_svr_loss_weight: 0.01
  v80_use_test_time_subset_search: true
  v80_ttvss_subset_sizes: [2, 3, 4, -1]   # -1 means full
  v80_ttvss_temperature: 1.0
```

### Training recipe

1. **Warm-start** from the best v57 checkpoint.
2. **Freeze** v25/v52/v57 for 1 epoch; train only the v80 VRBT head.
3. **Unfreeze** v52/v57 and fine-tune end-to-end with the v80 loss.
4. Apply v46-style random view dropout (`p=0.3`) and domain-balanced sampling across MPI-INF-3DHP / Shelf / Campus.
5. Validate with `eval_variable_views.py --view_subsets 2,3,4,full`.

### Success criteria

- Smoke test runs on RTX 4090 with no NaN/OOM and finite `MPJPE@k`.
- `MPJPE@2` improves by ≥10% over v57 on MPI-INF-3DHP val.
- Full-view `MPJPE@full` is within 1 mm of v57.
- Ablations isolate the contribution of VRBT and TTVSS.

---

## Paper story fit

v80 sharpens the paper claim: *Our multi-view pose estimator is robust to sparse and unreliable views, closing the accuracy gap between 2-view ad-hoc capture and dense 14-view studio capture.* By anchoring the model in true-3D datasets and explicitly reasoning about per-view reliability, the contribution becomes a reproducible robustness result rather than a chase for an inflated MPJPE record.

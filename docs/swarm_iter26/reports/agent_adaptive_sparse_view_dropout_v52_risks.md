# Agent Risk Report: v52 Adaptive Sparse-View Dropout

## Module

`adaptive_sparse_view_dropout_v52` — a learned, per-(view, joint) dropout gate that replaces the fixed-rate v46 view dropout.

## Risks and Mitigations

| ID | Risk | Concrete Symptom | Likelihood | Mitigation |
|----|------|------------------|------------|------------|
| R1 | **Straight-through top-K gradient bias** | Training loss plateaus; sparse-view `MPJPE@k` does not improve or degrades compared with v46. | Medium | Anneal the Gumbel-Softmax temperature `v52_asvd_temperature` from `1.0` to `0.1`; add a small entropy regularizer on the keep scores `p_vj` so gradients stay informative. |
| R2 | **Budget collapses to `min_views`** | `MPJPE@full` regresses because the model drops useful views even with all cameras available. | Medium | Initialize the budget MLP to select all views; freeze ASVD for `v52_asvd_warmup_epochs` so the baseline is learned first. |
| R3 | **Double masking with v51/v46 reliability** | Reliability head and dropout gate both down-weight the same views, leaving too little signal for triangulation. | Medium | Treat v51/v46 as *soft weights* and ASVD as a *hard selector*; feed reliability into ASVD but do not multiply by the same reliability again after masking. |
| R4 | **Inference/training budget mismatch** | Smoke val looks good, but fixed-budget evaluation at `k=2,3,4` fails because the learned budget differs from the eval setting. | Low | Expose an runtime `max_views` override; during eval disable stochastic sampling and use deterministic top-K. |
| R5 | **Domain-specific overfitting** | ASVD learns to keep only the camera layout seen during training, hurting 3DPW/in-the-wild generalization. | Medium | Pair ASVD with v51 domain conditioning; apply the dropout mask after domain-agnostic feature normalization, not before. |

## Recommended Smoke Test

Run `configs/benchmark_v52_adaptive_sparse_view_dropout_smoke.yaml` on the RTX 4090 and verify:

1. `val_MPJPE` at full views is within `1 mm` of the v51 baseline.
2. `MPJPE@3` and `MPJPE@4` are no worse than the v46 baseline.
3. No NaN/OOM and the average kept view count stays above `v52_asvd_min_views`.

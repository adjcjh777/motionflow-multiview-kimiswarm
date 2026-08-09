# v37: Self-Critique View Reliability Estimator

## Summary

Build on top of v34/v35/v36 by adding a **learned per-(view, joint) reliability score** that the model predicts from its own refined tokens. The reliability is used to soft-weight triangulation, cross-view attention, and reprojection losses. This is a self-evolution step: the model critiques its own per-view predictions and down-weights corrupted views without hard thresholds.

## Motivation

- v33 outlier-view rejection requires supervised outlier labels and a fixed z-threshold.
- v36 uncertainty gate is feature-level and inside the iterative graph block.
- Remaining errors often come from a single bad view (occlusion, calibration drift, motion blur) polluting the fused 3-D pose.
- A self-critique reliability estimator is fully differentiable, unsupervised, and can be learned from reprojection/temporal/bone-length residuals.

## Architecture

```text
v5 tokens (B, T, V, J, d)
  -> SelfCritiqueViewReliabilityV37
       -> per-view-joint reliability r(B, T, V, J) in [0, 1]
       -> per-view reliability r_view(B, T, V) in [0, 1]
  -> multiply into triangulation weights, attention scores, reprojection loss
```

## Implementation plan

1. New module `motionflow_mv/fusion/self_critique_view_reliability_v37.py`.
2. Class `SelfCritiqueViewReliabilityV37`:
   - Inputs: refined tokens `(B, T, V, J, d)`, optional raw 2-D points, camera params.
   - Outputs: `reliability (B, T, V, J)`, `view_reliability (B, T, V)`.
   - Architecture: 2-layer MLP over per-view-joint token + optional temporal 1-D conv + sigmoid.
   - Initialize output bias to ~0.9 so the model starts near identity.
3. Wire into `omniview_fusion_v5.py` after `v36_ugigr` / before triangulation head.
4. Add CLI flags:
   - `--use_self_critique_view_reliability_v37`
   - `--v37_scvr_hidden 64`
   - `--v37_scvr_n_layers 2`
   - `--v37_scvr_use_temporal_context`
   - `--v37_scvr_loss_weight 0.01`
5. Auxiliary self-supervised losses:
   - **Reprojection residual**: high reprojection error => low reliability.
   - **Temporal inconsistency**: large 2-D temporal jump => low reliability.
   - **Bone-length inconsistency**: inconsistent bone length => low reliability.
   - **Entropy regularization**: encourage reliability not to collapse to uniform.

## Training

- Warm-start from v36 checkpoint or train from scratch.
- Use the same WebBridge/H36M/MPI mixed loader.
- Smoke on RTX 4090 (clip_len=9, train_samples=100).
- Full run locally and on A800.

## Success criteria

- Smoke val_MPJPE < 30 mm (vs v36 smoke ~100 mm with tiny samples).
- Full run val_MPJPE < 27.08 mm (v35 baseline).
- Ablations show that down-weighting corrupted views improves robustness on outlier-augmented validation.

## Risks

- Reliability may collapse to a constant if auxiliary losses are too weak.
- Additional MLP adds parameters but small (~1% of model).
- Requires careful initialization and loss weight tuning.

## Related work in repo

- v33 outlier-view rejection: `motionflow_mv/fusion/outlier_view_rejection_v33.py`
- v36 UGIGR: `motionflow_mv/fusion/uncertainty_gated_iterative_graph_refinement_v36.py`
- Physical / reprojection losses in `experiments/train_omniview_fusion_v5_webbridge_multi.py`

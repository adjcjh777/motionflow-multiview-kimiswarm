# v44 Decision Plan

This document describes the candidate v44 architectures, conditional on the outcome of the A800 runs queued in issue #154.

## Pending A800 runs

| Priority | Run | Question it answers |
|---|---|---|
| 1 | `v25_geometry_fusion_all_train_baseline` | How strong is v25 on the full WebBridge mixed manifest? |
| 2 | `v25_geometry_fusion_all_train_plus_physical_domain` | Can physical loss + domain weights improve v25? |
| 3 | `v42_v36_physical_domain_no_v37` | Does the complex stack work on expanded data? |
| 4 | `v43_adaptive_node_residual_on_v42` | Does adaptive per-node residual help? |
| 5 | `v43_adaptive_node_residual_scaled` | Does capacity help the complex stack? |
| 6 | `v43_adaptive_node_residual_all_train` | Does more data help the complex stack? |

## Decision branches for v44

### Branch A: v25 wins (most likely as of 2026-08-09)

If `v25_geometry_fusion_all_train_baseline` is the best (or within ~1 mm of best):

1. **Baseline v44** = v25 + `--use_skeleton_physical_loss_v40 --domain_loss_weights 1.0,1.5` (already queued).
2. If v25+physical+domain beats plain v25, v44 keeps those two additions.
3. If v25+physical+domain does not beat plain v25, v44 = plain v25 with longer training / SWA / larger capacity.
4. Optional v44 ablations:
   - v25 + outlier-view augmentation
   - v25 + variable-view training
   - v25 + learned depth triangulation vs vanilla triangulation
   - v25 + geometry attention ablation

### Branch B: v42 beats v25

If `v42_v36_physical_domain_no_v37` clearly beats v25 on A800:

1. Keep v31-v36 stack.
2. v44 adds stronger regularization to fight epoch-1 overfitting:
   - Higher weight decay
   - More dropout
   - Stochastic depth
   - SWA / EMA
3. Consider scaling capacity only if overfitting is controlled.

### Branch C: v43 beats v42

If `v43_adaptive_node_residual_on_v42` beats v42 by > 5% relative:

1. Keep the adaptive per-node residual.
2. v44 explores edge-type-aware gating (already implemented as v44 smoke).
3. Add v44 full A800 run.

### Branch D: capacity/data wins

If `v43_adaptive_node_residual_scaled` or `v43_adaptive_node_residual_all_train` is best:

1. Scale v25 (if v25 is close) or v43 (if complex stack is close).
2. Use d=128, n_st_layers=3, batch_size=16, clip_len=13, train_samples=10000.
3. Focus on preventing overfit after epoch 1.

## Immediate next actions

1. Wait for A800 v25/v42/v43 results.
2. Update `docs/results_snapshot_2026_08_09.md` with A800 results.
3. Choose branch A/B/C/D and implement v44.
4. Run v44 smoke on RTX 4090, then full on A800.

## Publication story

The emerging paper story is:

> Multi-view human pose estimation benefits from strong geometry-aware fusion more than from complex graph/attention stacks. A simple v25 geometry-fusion baseline, augmented with skeleton-aware physical constraints and domain-balanced training, achieves state-of-the-art results on the WebBridge/H36M/MPI mixed benchmark.

This is conditional on the A800 v25 results confirming the local/A800 trend.

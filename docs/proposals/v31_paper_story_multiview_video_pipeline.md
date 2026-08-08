# v31 Paper Story: End-to-End Multi-View Video Pipeline

## Problem statement

MotionFlow-MultiView has grown into a collection of incremental modules (v18 deformable attention, v25 geometry fusion, v26 temporal geometry, v27 UDP, v28 physical-space alignment, v29 hierarchical encoder, v29 physical loss, v30 hardened hierarchical encoder). For the ICRA/CVPR 2027 paper we need a single, coherent end-to-end pipeline that can be drawn as:

```
multi-view video  ->  per-view 2-D pose/geometry  ->  hierarchical multi-view fusion  ->  physical-space alignment  ->  3-D human pose
```

The current best components are the v30 stable hierarchical encoder and the v29 physical-space temporal loss, but they have not been combined in a clean, full-capacity, TTE-free run. v29a showed that the hierarchical encoder can reach 28.12 mm on epoch 1 but overfits rapidly; v30 added stochastic depth and gated cross-scale fusion to combat this, and the physical loss needs a warmup so it does not dominate early training. TTE is broken and must stay disabled.

This proposal therefore defines a concrete v31 ablation, **paper_story_multiview_video_pipeline**, that integrates these validated pieces into one runnable training run.

## Concrete proposed change

Run the paper-story pipeline as a single training run with the following design choices:

- **Input representation**: calibrated multi-view video clips as `(points_2d, confidences, K, R, t)` from the WebBridge + H36M + MPI mixed loader.
- **Per-view embedding and aggregation**: keep `--use_camera_view_embedding` and `--use_set_view_aggregator` so the model is permutation-invariant over an arbitrary number of views.
- **Hierarchical multi-view fusion**: enable v30 encoder with stochastic depth regularization:
  - `--use_hierarchical_multiview_v30`
  - `--v30_n_part_layers 2`
  - `--v30_stochastic_depth_prob 0.1`
- **Variable-view robustness**: keep the variable-view curriculum (`--use_variable_view_training`) with `min_views=2`, `max_views=14`, starting from 4 views and alpha=2.0, plus `--variable_view_permute`. This preserves the paper claim of handling arbitrary view subsets.
- **Out-of-distribution robustness**: keep outlier-view augmentation (`--outlier_view_prob 0.3`).
- **Physical-space alignment**: enable v29 physical loss with a warmup so it ramps in smoothly:
  - `--use_physical_space_temporal_loss_v29`
  - `--v29_floor_loss_weight 0.01`
  - `--v29_bone_temporal_weight 0.01`
  - `--v29_com_jitter_weight 0.001`
  - `--v29_physical_loss_warmup_epochs 2`
- **What is intentionally left out**: no TTE module (`--use_test_time_self_evolution_v29` is off), because TTE is broken and produces ~90 mm validation errors.

Full-capacity training uses `d=128`, `residual_hidden=256`, `n_st_layers=3`, `batch_size=16`, `train_samples=4000`. A smaller smoke variant uses `d=64`, `batch_size=4`, `train_samples=200` for fast local validation on the RTX 4090.

## Expected impact on `val_MPJPE` / overfitting

- **val_MPJPE**: We expect the paper-story run to reach ≤28 mm on the first validation (matching v29a) and, crucially, to remain stable or improve through epoch 5 instead of overfitting to >80 mm. The v30 hardening (stochastic depth, gated residuals, dataset-aware part groups) should suppress the overfitting seen in v29a.
- **Overfitting**: Stochastic depth acts as a strong regularizer on the cross-view attention, and the physical loss adds an inductive bias toward temporally consistent, physically plausible poses. The two-epoch warmup prevents the physical loss from distorting the early geometry-learning phase.
- **Physical plausibility**: Compared to runs without physical loss, we expect lower floor penetration, smoother bone-length variation, and reduced center-of-mass jitter on long sequences.

## Main risk

The main risk is **negative interaction between the hierarchical encoder and the physical loss schedule**. If the physical loss warms up too slowly, the v30 encoder may still overfit before the regularizer becomes active; if it warms up too fast, it may pull early poses toward an incorrect physical prior and slow convergence. The smoke run is intended to catch this quickly before committing A800-D resources.

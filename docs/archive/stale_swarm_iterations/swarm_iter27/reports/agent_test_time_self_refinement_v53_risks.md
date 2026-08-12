# v53 Test-Time Self-Refinement: Risks and Mitigations

## 1. Inference latency from iterative unrolling

**Risk.** v53 is designed to run `v53_ttsr_num_steps` refinement steps at inference. Each step re-computes reprojection residuals and a graph-network pass. With `num_steps=3` and the skeleton-graph operating over `J=17` joints, the module can add 15–30% to per-batch latency on A800 and become a bottleneck for real-time or high-throughput evaluation.

**Mitigation.**
- Default `v53_ttsr_num_steps=1` for training; use `num_steps=3` only at inference.
- Benchmark latency on the smoke run before the full A800 launch.
- Provide a fast-path flag `v53_ttsr_use_single_step_eval=True` that runs only one step if latency is constrained.
- Cache the projection matrices `K, R, t` and reuse them across steps to avoid redundant tensor operations.

## 2. Gradient instability through reprojection residuals

**Risk.** The correction head uses reprojection residuals, whose Jacobian with respect to the 3-D pose contains division by depth. When a joint is far from the camera or nearly parallel to the image plane, small pose perturbations can produce exploding or vanishing gradients, destabilizing early training.

**Mitigation.**
- Clamp depth to `z ≥ 1e-6` and use a Huber-smoothed reprojection loss rather than raw squared error.
- Stop gradients through the residual feature when `v53_ttsr_num_steps > 1` in training, so the network learns from the residual *value* but not its full Jacobian chain.
- Initialize the correction head to zero and ramp the loss weight with `v53_ttsr_warmup_epochs` (default `1`), so gradients start tiny.

## 3. Double-counting with v27/v29 test-time self-evolution

**Risk.** `OmniMultiViewFusionV5` already has optional test-time modules (`use_test_time_self_evolution_v27` and `v29`) that refine poses at evaluation. Adding v53 after them could apply two similar corrections, either cancelling each other out or over-sharpening the pose and amplifying noise.

**Mitigation.**
- Disable v27/v29 by default when v53 is enabled, or gate with a runtime assertion that at most one test-time refinement path is active.
- In the first smoke, run four configurations: baseline, v27 only, v53 only, and v27+v53 together, and compare MPJPE and reprojection residuals.
- Treat v53 as the canonical test-time path and deprecate v27/v29 for new experiments.

## 4. Overfitting to the residual-MLP output manifold

**Risk.** Because v53 is inserted after the final residual MLP, its inputs are highly processed. The self-refinement network may learn to memorize residual-MLP errors rather than true multi-view geometry, causing it to degrade on out-of-distribution domains or sparse views.

**Mitigation.**
- Strongly regularize the graph network with dropout (`v53_ttsr_dropout=0.1`) and weight decay (`1e-4`).
- Use bone-length and temporal self-supervised losses as inductive biases so the correction depends on physical/geometric structure, not just residual-MLP patterns.
- Validate on a held-out 3DPW actual split and on the v46 `MPJPE@2` task before committing to a full A800 run.

## 5. Identity-at-init failure due to non-zero batch-norm / layer-norm statistics

**Risk.** If the graph network uses LayerNorm or BatchNorm, the zero-initialized output head still passes through normalized activations, and the module may not be a perfect identity at initialization. This would break the warm-start property promised for v53.

**Mitigation.**
- Use residual connections around the graph layers and omit normalization on the final correction/gate heads.
- Add an unit test that loads a pretrained v52 checkpoint with `use_v53_test_time_self_refinement=True` and asserts `||pred_3d_v53 - pred_3d_v52|| < 1e-3` mm with `loss_weight=0` and `num_steps=1`.
- If normalization is required for stability, initialize normalization affine parameters to identity and freeze them for the first epoch.

# v54 Test-Time Self-Refinement: Risks and Mitigations

## 1. Inference latency from iterative unrolling

**Risk.** v54 is designed to run `v54_ttsr_num_steps` refinement steps at inference, and each step re-computes reprojection residuals, v52/v53 physical features, and a skeleton-graph pass. With `num_steps=3` and a graph network over `J=17` joints, the module can add 15–30% to per-batch latency on A800 and become a bottleneck for real-time or high-throughput evaluation.

**Mitigation.**
- Default `v54_ttsr_train_steps=1` during training; use `num_steps=3` only at inference.
- Benchmark latency on the smoke run before the full A800 launch.
- Provide a fast-path flag `v54_ttsr_use_single_step_eval=True` that runs only one step when latency is constrained.
- Cache projection matrices `K, R, t`, the v52 weight tensor, and the v53 floor/bone features across steps to avoid redundant tensor operations.

## 2. Gradient instability through reprojection residuals

**Risk.** The correction head uses reprojection residuals, whose Jacobian with respect to the 3-D pose contains division by depth. When a joint is far from the camera or nearly parallel to the image plane, small pose perturbations can produce exploding or vanishing gradients, destabilizing early training.

**Mitigation.**
- Clamp depth to `z ≥ 1e-6` and use a Huber-smoothed reprojection loss rather than raw squared error.
- Stop gradients through the residual feature when `v54_ttsr_num_steps > 1` in training, so the network learns from the residual *value* but not its full Jacobian chain.
- Initialize the correction head to zero and ramp the loss weight with `v54_ttsr_warmup_epochs` (default `1`), so gradients start tiny.

## 3. Over-constraining the v53-calibrated pose

**Risk.** v54 is inserted immediately after v53 Physical-Space Calibration, which already applies floor, bone-length, and reprojection corrections. Adding another refinement loop on top can over-constrain the pose, pulling joints away from the true multi-view evidence or amplifying v53-specific biases.

**Mitigation.**
- Use a small `v54_ttsr_step_size` (default `0.1`) and a gated residual initialized near zero (`residual_gate_init=-6.0`) so corrections start small and grow only where needed.
- Treat v53 outputs as *features*, not hard constraints: the v54 loss only applies its own physical terms with moderate weights (`v54_ttsr_floor_weight=0.01`, `v54_ttsr_bone_weight=0.1`).
- Run a smoke ablation with `v53` only, `v54` only, and `v53+v54` to verify the combination improves rather than degrades the baseline.

## 4. Overfitting to the v53 output manifold

**Risk.** Because v54 receives the output of v53, its inputs are already heavily processed. The self-refinement network may memorize v53 errors rather than true multi-view geometry, causing degradation on out-of-distribution domains or sparse views.

**Mitigation.**
- Regularize the graph network with dropout (`v54_ttsr_dropout=0.1`) and weight decay (`1e-4`).
- Include raw reprojection and bone-length signals as explicit features so the correction depends on geometry, not only on v53 activations.
- Validate on a held-out 3DPW actual split and on the v46 `MPJPE@2` task before committing to a full A800 run.

## 5. Identity-at-init failure due to non-zero physical hints

**Risk.** The module promises identity-at-init, but v53 floor/bone features may be non-zero even at initialization. If the graph network does not preserve the zero residual path, the output may differ from the input at initialization, breaking the warm-start property.

**Mitigation.**
- Zero-initialize the final correction and gate projection layers, and initialize the residual gate logit to `-6.0` so `sigmoid(gate) ≈ 0.0025`.
- Residual-connect the graph layers and omit normalization on the final correction/gate heads.
- Add a unit test that loads a pretrained v53 checkpoint with `use_v54_test_time_self_refinement=True` and asserts `||pred_3d_v54 - pred_3d_v53|| < 1e-3` mm with `loss_weight=0` and `num_steps=1`.

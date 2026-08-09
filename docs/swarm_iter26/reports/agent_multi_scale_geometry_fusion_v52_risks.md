# v52 Multi-Scale Geometry Fusion — Risks and Mitigations

## Risk 1: Part-group mismatch across datasets

**Description:** The part-scale branch relies on a fixed anatomical grouping (e.g., H36M 17-joint groups). MPI-INF-3DHP uses 28 joints and WebBridge may use a different skeleton layout; a hard H36M grouping will produce meaningless pooled tokens on those datasets and can hurt generalization.

**Mitigation:**
- Make `v52_msgf_part_groups` configurable per dataset in the YAML.
- Implement the grouping via a learnable soft assignment matrix `A` with a temperature-controlled softmax. Initialize `A` with the dataset-specific hard grouping, then let training sharpen or soften it.
- Add a smoke test that runs the module on each supported skeleton layout and checks that pooled tokens have finite, non-zero variance.

## Risk 2: Geometry bias amplifies camera calibration errors

**Description:** The ray-intersection and epipolar biases assume accurate intrinsics/extrinsics. If principal-point correction or extrinsics are noisy, the bias can push attention toward incorrect view pairs and degrade triangulation.

**Mitigation:**
- Initialize the geometry-gain scalar `γ_s` to 0.0 so the bias is ignored at the start.
- Clamp the raw epipolar and ray-intersection values to a bounded range (e.g., `[-10, 10]`) before adding them to attention logits.
- Make `γ_s` a learned scalar gated by a confidence head that predicts how much to trust the geometric bias; initialize that confidence head to output near zero.

## Risk 3: Compute and memory overhead from multi-scale attention

**Description:** Running cross-view attention at three scales multiplies memory and runtime, especially because the body and part scales still attend over all `V` views. On the RTX 4090 smoke test this could OOM or slow iteration times.

**Mitigation:**
- Keep `v52_msgf_n_layers=1` and `v52_msgf_hidden=64` by default.
- Pool tokens before attention at the part and body scales; for example, 6 part tokens + 1 body token instead of `J` joint tokens per view.
- Add an optional `v52_msgf_stochastic_depth_prob` that randomly drops the whole block during training, and profile a single forward pass before launching a full smoke run.

## Risk 4: Interaction with existing regularization modules

**Description:** v52 is added on top of v45 adaptive geometry fusion, v46 sparse-view generalization, v47/v49 temporal aggregation, and v48 domain generalization. Stacking another regularizer may over-constrain the model, causing underfitting or unstable training if all gates are active simultaneously.

**Mitigation:**
- Default `v52_msgf_aux_loss_weight=0.0` so no extra loss is added initially.
- Train smoke variants both with and without v45/v46 active; compare val_MPJPE to isolate v52’s contribution.
- If stacking causes instability, gate v52 on only when `v46_reliability` is below a threshold (i.e., use v52 as a fallback for uncertain views).

## Risk 5: Warm-start / identity-at-init is not actually achieved

**Description:** Despite zero-initialized final layers, subtle sources of non-identity behavior (e.g., LayerNorm inside the block, learned scale assignment, or broadcasted positional embeddings) could perturb pre-trained features before training has even updated the new parameters.

**Mitigation:**
- Implement a deterministic unit test that instantiates the module and asserts `torch.allclose(output, feat, atol=1e-5)` at initialization for random inputs.
- If LayerNorm is used, initialize its affine parameters to `weight=1, bias=0` and place it inside the residual branch only, never on the shortcut path.
- Document the warm-start invariant in the module docstring and add a CI test that fails if it is violated.

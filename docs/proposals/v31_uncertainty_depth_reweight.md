# v31: Uncertainty-Aware Depth Reweighting for Multi-View Geometry Fusion

## Problem statement

v27's `UncertaintyDepthProposalTriangulation` predicts a per-ray Gaussian depth distribution (`mu`, `sigma`, `mix_weights`) and samples depth hypotheses during training. However, the final 3D estimate is fused by a softmax over `(view, sample)` candidate scores that does **not** use the predicted uncertainty: a high-σ (uncertain) ray has the same softmax capacity as a low-σ (certain) one. Noisy views or joints are therefore not explicitly down-weighted, which limits robustness and contributes to the overfitting seen in v29a and the v30 hierarchical encoder.

## Concrete proposed change

Add an **uncertainty reweighting gate** inside `UncertaintyDepthProposalTriangulation`:

1. From the predicted per-ray mixture, compute the mixture-weighted depth standard deviation `σ_bar(v, j)`.
2. Convert it to a certainty weight `w_σ = 1 / (1 + σ_bar)`.
3. Modulate the candidate scores before the `(V, S)` softmax:
   ```
   s' = s * (1 + λ_σ * w_σ)
   ```
   where `λ_σ` is a learned scalar initialised to `0.0`.
4. Keep the v27 uncertainty regularisation loss, `min_sigma` clamp, and `residual_scale` identity-at-init property intact.

At `λ_σ = 0` the block is exactly the current v27, so warm starts are preserved. No new CLI flag is required; the change is gated by `--use_uncertainty_depth_proposals_v27`.

## Expected impact

- **val_MPJPE:** small but consistent improvement (a few millimetres) on WebBridge/H36M mixed validation, because cleaner geometry seeds flow into the rest of the network.
- **Overfitting:** reduced. The geometry head becomes more data-efficient and less likely to latch onto confident-but-wrong proposals, which should help the v30 encoder's tendency to overfit after epoch 1.
- **Robustness:** improved handling of outlier views and occluded joints, since high-σ rays contribute less regardless of their raw softmax score.

## Main risk

- **Uncertainty collapse.** The depth-distribution MLP may drive all `σ` very small, making `w_σ` uniform and the reweighting ineffective. This is partly mitigated by the existing `min_sigma` clamp and v27 regularisation loss, but an additional diversity term on `w_σ` may be needed if collapse is observed.

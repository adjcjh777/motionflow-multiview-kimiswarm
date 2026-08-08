# v31: Epipolar-Guided Sampling

## Problem statement

The v30 hardened hierarchical encoder fuses views at joint, part, and body scales with dense cross-view attention. While this gives rich multi-scale context, every query view attends to every other view, so occluded or geometrically inconsistent views still contribute noise and the model is prone to overfitting after the first epoch (v29a rose from 28 mm to 81 mm across three epochs). The v18 deformable cross-view attention module already computes epipolar-line distances and biases the attention logits with them, but in the v30 smoke run it is disabled and its top-k straight-through path has not been evaluated together with v30’s stochastic-depth/gated-residual hierarchy. We therefore want to make cross-view fusion explicitly geometry-aware by **sampling key views from an epipolar consistency distribution** before the attention is even computed.

## Concrete proposed change

Introduce a new `EpipolarGuidedViewSampler` (v31) that sits between the camera-conditioned feature embedding and the v30 hierarchical encoder.

1. **Epipolar consistency score.** Re-use `compute_epipolar_distance` from `motionflow_mv/fusion/epipolar_attention_bias.py` to compute, for each joint, the symmetric epipolar-line distance between every pair of views.
2. **Hard top-k view sampling.** For each query view and joint, select the `k = max(2, V // 2)` source views with the smallest epipolar distance and build a binary key-view mask. This is the “sampling” step.
3. **Sparse cross-view attention.** The v30 hierarchical encoder’s joint/part/body attention blocks attend only over the sampled key views. Masked-out views are excluded from softmax normalization and gradients, reducing noise.
4. **Identity at init / warm start.** The sampler is wrapped in a gated residual whose scale is initialised near zero, so the network can fall back to the v30 baseline. The existing `--use_deformable_cross_view_attention_v18 --deformable_attention_use_topk_st` path is used as the immediate smoke-test proxy.
5. **Safety constraints.** The broken TTE module remains disabled. Physical loss is kept with the same warmup schedule used in v30; if it destabilizes the run, it is dropped.

Implementation hooks for a future source patch: add `--use_epipolar_guided_sampling_v31`, `--v31_top_k`, and `--v31_epipolar_temperature` flags to `experiments/train_omniview_fusion_v5_webbridge_multi.py` and `motionflow_mv/fusion/omniview_fusion_v5.py`.

## Expected impact on val_MPJPE / overfitting

* **val_MPJPE:** likely 1–3 mm better than the v30 smoke baseline on WebBridge/H36M because geometry-guided sampling suppresses inconsistent view pairs and lets the model focus on reliable rays.
* **Overfitting:** the hard top-k mask acts as a regulariser and reduces attention entropy, which should blunt the post-epoch-1 rise seen in v29a. Stochastic depth in v30 already drops blocks; adding view sampling further constrains the effective capacity.

## Main risk

If camera calibration is perturbed or the subject has large non-rigid deformation (loose clothing, fast motion), epipolar distance can misrank good views as inconsistent, especially if `k` is too small. The straight-through top-k estimator may also introduce biased gradients early in training. `k` and the epipolar temperature therefore need a small grid search; the smoke run uses a conservative `k = max(2, V // 2)` and monitors epoch-1 validation closely.

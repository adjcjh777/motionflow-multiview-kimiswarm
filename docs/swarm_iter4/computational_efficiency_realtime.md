# Computational Efficiency and Real-Time Requirements for MotionFlow

## Survey

Real-time multi-view human pose estimation is a hard constraint for many ICRA/CVPR applications—robotic assistance, sports analysis, and AR/VR all target 25–30 Hz end-to-end. Our current `RayAttentionFusionModel` (`motionflow_mv/fusion/ray_attention_model.py`) fuses V calibrated views per joint through a small transformer head, then triangulates with a differentiable weighted DLT layer. It is accurate, but the current implementation is not optimized for throughput. The forward pass (1) inverts intrinsics `K` per sample inside `_compute_rays`, (2) runs `torch.linalg.lstsq` once per joint in `_triangulate_weighted_dlt`, and (3) reshapes attention over `(B*J, V, d)`. These choices create avoidable per-batch overhead and make the model more expensive than the closed-form DLT baseline it aims to replace.

The training script `experiments/train_ray_attention_real.py` uses `batch_size=32`, `d=64`, 4 heads, and a 17-joint skeleton, but it does not measure or bound inference latency. For a real-time claim, we must show that the accuracy gain over DLT does not come at the cost of unusable runtime.

## Actionable Recommendations

1. **Cache and broadcast constant camera tensors.** The intrinsics `K` and the rotation `R` are fixed for a given rig, yet `_compute_rays` calls `torch.inverse(K)` inside every forward pass. Pre-compute `K_inv` once and store it alongside `K`, `R`, `t`, and the projection matrix `P`. The forward pass should only assemble per-frame 2D points and run the learned weight head.

2. **Vectorize the DLT solver across joints.** `_triangulate_weighted_dlt` loops over `J` and calls `torch.linalg.lstsq` once per joint. Stack the `A` matrices across joints into shape `(B*J, 2V, 4)` and call `torch.linalg.lstsq` once; this typically yields a 2–5× reduction in solver overhead for `J=17`. Keep a fallback per-joint loop only when batched `lstsq` fails on small or ill-conditioned inputs.

3. **Profile and optionally shrink the attention head.** With `d=64`, 4 heads, and `V≤4`, the attention cost is small, but it still dominates when compared to a plain DLT baseline. Add a latency-accuracy ablation with `d∈{16,32,64}` and `num_heads∈{1,2,4}`. If `d=32` or `d=16` reaches the same MPJPE, adopt the smaller head for the deployment model and report its FLOPs/latency.

4. **Add a real-time inference benchmark.** Create `experiments/bench_ray_attention_latency.py` that runs a warm-up, then times `N` forward passes on CPU and GPU with `batch_size=1` (streaming) and `batch_size=32` (batch). Report mean and P99 latency, throughput (FPS), and memory. Include the DLT baseline as a reference. A target to beat is 30 FPS per-subject on a mid-range GPU for `V=4`, `J=17`.

5. **Prepare a TensorRT/ONNX deployment path.** After the model is trained, export to ONNX and, if possible, TensorRT. Pay special attention to the `torch.linalg.lstsq` node: some runtimes do not support it, so add an export-time option to replace the weighted DLT with a fixed-iteration weighted least-squares solver (e.g., 5 conjugate-gradient steps) that maps cleanly to ONNX ops. This also protects against deployment failures on edge devices.

## Potential Risks

- **Numerical instability if `lstsq` is replaced naïvely.** A vectorized or iterative solver must preserve the weighting and handle rank-deficient cases (occluded/outlier views) as gracefully as the current looped `lstsq`.
- **Memory blow-up from `(B*J, V, d)` attention.** For large `J` or batch, the reshape multiplies memory by `J`. Use mixed-precision training/inference (`torch.cuda.amp`) and keep the deployment head small.
- **Accuracy-latency trade-off.** Shrinking `d` or replacing `lstsq` may increase MPJPE. Any deployment model must report both metrics; real-time claims cannot sacrifice the accuracy advantage over DLT.
- **Hardware assumptions.** CVPR/ICRA reviewers will ask about GPU/CPU and batch size. Latency numbers must include the hardware configuration and warm-up protocol.

## Fit into Paper Plan

This work belongs in a paper section on **Efficient Calibrated Fusion** or **Real-Time Multi-View Inference**. We will present a latency-accuracy Pareto curve that compares DLT, `ray_attention` at different `d`, and the ONNX/TensorRT exported model. The core narrative: the ray-aware attention head adds only a lightweight per-joint weighting step on top of a geometrically-correct triangulator, and with the optimizations above it runs at real-time rates while improving robustness to occlusion and noisy 2D detections.

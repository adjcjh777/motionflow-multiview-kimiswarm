# Multi-scale temporal fusion

I investigated multi-scale temporal fusion in the project. I do not have write access to disk in this read-only exploration role, so the report content is below and should be saved to the requested path by the parent agent.

**Report path:** `docs/swarm_iter7/multi_scale_temporal_fusion.md`

---

# Multi-Scale Temporal Fusion

## 1. Current state

- `motionflow_mv/fusion/multiscale_temporal_conv_model.py:32-79` — `MultiScaleTemporalBlock` already implements parallel 1-D temporal convolutions with configurable kernel sizes and dilations.
- `motionflow_mv/fusion/multiscale_temporal_conv_model.py:82-279` — `MultiScaleTemporalConvModel` keeps the v3 per-frame ray/view/joint encoder and replaces the temporal transformer with a stack of these multi-scale conv blocks.
- `tests/test_multiscale_temporal.py` — forward/backward shape checks and custom kernel/dilation tests.
- `experiments/train_multiscale_temporal_mpiinf3dhp.py` — training script for MPI-INF-3DHP.
- `docs/swarm_iter5/multiscale_temporal_conv.md` — prior report: on a 250-frame smoke subset the conv head matched the temporal transformer baseline within 0.02 mm (29.96 vs 29.94 mm).
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py:38-185` — current best model `RayAttentionFusionModelTemporalResidual` uses a **temporal transformer** + residual refinement head; achieves 11.17 mm MPJPE / 8.24 mm PA-MPJPE on MPI-INF-3DHP and 5.74 mm / 3.99 mm on Human3.6M (`docs/swarm_iter7/exploration_summary.md`).

## 2. Gap / opportunity

The multi-scale temporal conv head has not yet been combined with the residual refinement head. If dilated convolutions can replace the temporal transformer in the current best architecture, the paper gains:

- A simpler, potentially faster temporal backbone.
- A clean ablation: transformer vs. multi-scale conv temporal fusion.
- A path to a lighter model with competitive accuracy.

## 3. Concrete next step

Create `motionflow_mv/fusion/multiscale_temporal_residual_model.py` by:

1. Reusing `MultiScaleTemporalBlock` from `multiscale_temporal_conv_model.py`.
2. Wrapping it around the per-frame ray-attention encoder and weight head from `RayAttentionFusionModelTemporal`.
3. Adding the residual MLP head from `RayAttentionFusionModelTemporalResidual`.
4. Writing/adapting a trainer (e.g., `experiments/train_multiscale_temporal_residual_mpiinf3dhp.py`) using the same MPI-INF-3DHP S1→S2/Seq1 protocol: `clip_len=13`, `d=64`, `residual_hidden=128`, same seed/data as `train_ray_attention_temporal_residual_mpiinf3dhp.py`.
5. Running `experiments/benchmark_residual_temporal.py` to compare throughput against the current final5 model.

## 4. Expected success metric

- MPI-INF-3DHP cross-subject MPJPE ≤ 11.17 mm (match or beat `ray_attention_temporal_residual_final5`).
- Human3.6M cross-subject MPJPE ≤ 5.74 mm, PA-MPJPE ≤ 3.99 mm.
- Inference throughput ≥ current final5 on the benchmark.
- If the conv backbone underperforms, document whether larger kernels/dilations or a hybrid transformer+conv block helps.

## 5. Risks / blockers

- **A800-D / Docker are read-only**; any new training must run locally or on accessible WSL hardware.
- **WebBridge MPI-INF-3DHP data** may need to be downloaded; do not commit large `.npz` checkpoints.
- **Limited receptive field** — pure convs may miss long-range temporal dependencies unless dilations/kernels are scaled or combined with attention.
- **Code duplication** — `multiscale_temporal_conv_model.py` duplicates the per-frame encoder; the new model should avoid further duplication, preferably by factoring out the shared encoder.

---
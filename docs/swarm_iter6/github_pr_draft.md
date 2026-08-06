# Add residual refinement head to MotionFlow-MultiView fusion

## Summary
This PR adds a temporal ray-attention fusion model with a lightweight residual refinement head, enabling MotionFlow to fuse calibrated multi-view video into metric 3D human pose.

## Key changes
- New `RayAttentionFusionModelTemporalResidual` in `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`.
- New `FusionModule` plugin in `motionflow_mv/fusion/ray_attention_temporal_residual_module.py`.
- Training script `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`.
- Evaluation script `experiments/eval_ray_attention_temporal_residual_v3.py` reporting MPJPE, PA-MPJPE, PCK, and AUC.
- Benchmark script `experiments/benchmark_residual_temporal.py` for RTX 4090 latency/throughput.
- Paper-level visualizations and a 6-page draft in `docs/`.

## Verified results
- MPI-INF-3DHP cross-subject: **11.17 mm** MPJPE / **8.24 mm** PA-MPJPE (243 k params, `d=64, h=128`, 5 epochs).
- MPI-INF-3DHP lightweight: **13.22 mm** MPJPE / **11.77 mm** PA-MPJPE (66 k params, `d=32, h=64`).
- Human3.6M cross-subject: **5.74 mm** MPJPE / **3.99 mm** PA-MPJPE.
- RTX 4090: 12.8–194.8 clips/s depending on batch size.

## Testing
```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v3.py \
    --checkpoint outputs/ray_attention_temporal_residual_mpi_d32_h64.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 32 --residual_hidden 64 --batch_size 8
```

## Checklist
- [x] Core model and plugin implemented
- [x] Training / evaluation scripts added
- [x] MPI-INF-3DHP and Human3.6M validated
- [x] Runtime benchmarked
- [x] Paper skeleton drafted
- [ ] `gh` CLI installed and authenticated (needs maintainer)
- [ ] Real-world GVHMR demo added (follow-up PR)

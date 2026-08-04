# [Exploration] Residual refinement head for MotionFlow-MultiView

## Motivation
MotionFlow currently operates on monocular video. Real-world capture, however, is overwhelmingly multi-view. This issue tracks the exploration of a calibrated multi-view extension of MotionFlow, with a temporal ray-attention fusion model and a lightweight residual refinement head.

## Current status
- Implemented `RayAttentionFusionModelTemporalResidual` as a `FusionModule` plugin.
- Verified on MPI-INF-3DHP and Human3.6M.
- Best verified numbers:
  - MPI-INF-3DHP (S1→S2/Seq1): **11.17 mm** MPJPE / **8.24 mm** PA-MPJPE, 243 k parameters (`d=64, h=128`, 5 epochs).
  - MPI-INF-3DHP lightweight (`d=32, h=64`): **13.22 mm** MPJPE / **11.77 mm** PA-MPJPE, 66 k parameters.
  - Human3.6M (S1→S5): **5.74 mm** MPJPE / **3.99 mm** PA-MPJPE.
- Generated qualitative figures, runtime benchmark on RTX 4090, and a 6-page paper skeleton.

## Next steps
- [ ] Install `gh` CLI and authenticate to open tracking issues / PRs.
- [ ] Merge the residual refinement plugin into the main MotionFlow pipeline.
- [ ] Add real-world GVHMR output demo.
- [ ] Run full ablations (residual head capacity, temporal window, occlusion robustness).
- [ ] Produce camera-ready figures for ICRA / CVPR 2027.

## Relevant files
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_module.py`
- `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`
- `experiments/eval_ray_attention_temporal_residual_v3.py`
- `docs/paper_draft_icra_cvpr_2027.md`

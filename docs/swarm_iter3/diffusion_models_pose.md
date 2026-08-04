Research note written to `docs/swarm_iter3/diffusion_models_pose.md`.

Summary of what the report covers:

- **Problem**: The current MotionFlow multi-view pipeline is led by geometric DLT (~10 px reproj. error), while learned plugins lag (~80 px). Diffusion models can inject strong SMPL/SMPL-X priors, uncertainty, multi-view consistency, and pseudo-labels.
- **Key works cited**: ScoreHMR (CVPR 2024), Mocap-2-to-3 (arXiv 2025), RapidPoseTriangulation (arXiv 2025), GPS-Gaussian+, HumanSplat, PSHuman, plus AMASS.
- **Codebase fit**: `scorehmr_adapter.py`, the `FusionModule` plugin registry, `attention_model_v2.py`, and `temporal_refiner.py` are all natural integration points.
- **Recommendations**:
  1. Build a `ScoreHMRMultiViewFusion` plugin that uses calibrated multi-view reprojection score guidance.
  2. Use ScoreHMR to generate pseudo-3D labels when real 3D GT is unavailable.
  3. Replace/augment the Bi-GRU temporal refiner with a small diffusion motion prior on AMASS.
  4. Optionally add neural rendering regularization.
- **Risks**: real-time inference cost, calibration/metric-scale alignment, 3D GT validation, licensing, uncertainty propagation, and A800-D access constraints.

The file is 103 lines and ready for review.
# Literature Gap Survey — Summary

**Produced:** 2026-08-04 in `docs/swarm_iter5/literature_gap.md`

**What was done:**
Surveyed CVPR 2025 (open-access proceedings), arXiv 2025-2026 pre-prints, and the project's prior swarm positioning docs. Focused on calibrated/weakly-calibrated multi-view human pose/motion estimation.

**Most important findings:**
1. **Geometry-aware learned fusion is the new baseline** — ray/camera embeddings plus differentiable triangulation are replacing both vanilla attention and pure geometry.
2. **Cross-calibration generalization is under-reported** — most 2025-2026 papers train and test on the same rig; real robots need cross-rig robustness.
3. **Uncertainty and plug-in operation are missing** — few multi-view pose methods export per-joint uncertainty or run as a lightweight interchangeable module.
4. **Temporal consistency and parametric body recovery are adjacent frontiers** — adding bone-length/temporal losses and a multi-view SMPL fit would broaden the paper.
5. **ICRA angle is strong** — the robotics community needs metric, uncertainty-aware pose fusion more than it needs another benchmark-tuned pose estimator.

**Immediate opportunities for motionflow-multiview:**
- Run cross-dataset ablations (train Shelf → test Campus/H36M) with `ray_attention_v3`.
- Add bone-length + temporal losses to the v3 trainer.
- Extend `HumanMotionIR` with per-view weights and reprojection residuals.
- Keep the model lightweight; avoid diffusion/SSM unless v3 clearly saturates.

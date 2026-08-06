Completed the report at `docs/swarm_iter3/icra2027_paper_positioning.md`.

**Brief summary:**

- **Positioning:** Frame the work as a *modular, uncertainty-aware, geometry-guided multi-view human motion fusion framework* rather than a pure learned triangulator. The core story is studying **where and how learning improves over DLT**, validated on real 3D benchmarks and robot downstream tasks.
- **Key finding from the repo:** On Shelf (frames 300–600), DLT is ~9.9/5.5 px mean/median reprojection error. All learned plugins trained only on reprojection loss or DLT pseudo-GT merely tie DLT, confirming that **reprojection-only supervision cannot beat triangulation**.
- **Top recommendations:**
  1. Switch to real 3D-supervised training (Human3.6M, Panoptic, 3DPW) with `L_MPJPE + reprojection + bone-length + temporal + symmetry` losses.
  2. Adopt an **MVGFormer-style** geometry + appearance fusion (DLT/MVGFormer → learned residual).
  3. Add a **HeatFormer-style** skeleton-to-SMPL refiner for parametric robot-ready output.
  4. For **ICRA**: validate on robot retargeting/policy metrics (foot sliding, end-effector error, policy success).
  5. For **CVPR**: populate `HumanMotionIR.uncertainty` with per-view weights, reprojection residuals, and per-joint std.
- **Main risks:** 3D GT access, reprojection-only ceiling, calibration sensitivity, and per-view estimator cost.